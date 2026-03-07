"""
Tool Execution Sandbox for NeuroCore

Provides a secure execution environment for custom Python tools with:
- Restricted builtins and globals
- Resource limits (timeout, memory, output size)
- Network access controls
- Filesystem restrictions
"""

import builtins
import sys
import os
import re
import signal
import threading
import functools
from typing import Any, Dict, Set, Optional, List
from contextlib import contextmanager
import json
from multiprocessing import Process, Queue
import logging
import tracemalloc

logger = logging.getLogger(__name__)

# Store the real __import__ function at module load time
# This avoids issues where __builtins__ might be a module instead of a dict
_real_import = builtins.__import__

# Memory limit enforcement - platform specific
try:
    import resource
    HAS_RESOURCE_MODULE = True
except ImportError:
    HAS_RESOURCE_MODULE = False




# Dangerous builtins that should never be available in sandboxed code
DANGEROUS_BUILTINS: Set[str] = {
    'open', 'input', 'raw_input', 'exec', 'eval', 'compile', 
    'breakpoint', 'help', 'quit', 'exit', '__import__',
    'reload', 'file', 'execfile',
}

# Dangerous modules that should not be importable
DANGEROUS_MODULES: Set[str] = {
    'os', 'sys', 'subprocess', 'socket', 'multiprocessing',
    'threading', 'ctypes', 'mmap', 'resource', 'signal',
    'pickle', 'cPickle', 'marshal', 'imp', 'importlib',
    'site', 'warnings', 'traceback', 'gc', 'inspect',
    'types', 'code', 'codeop', 'pdb', 'bdb', 'cmd',
    'shutil', 'tempfile', 'pathlib', 'path', 'glob',
    'fnmatch', 'linecache', 'tracemalloc', 'faulthandler',
    'posix', 'nt', 'java', 'org', 'com', 'net',
}

# Allowed safe modules
SAFE_MODULES: Set[str] = {
    'math', 'random', 'datetime', 'time', 'calendar',
    'decimal', 'fractions', 'numbers', 'statistics',
    'itertools', 'functools', 'operator', 'copy',
    'string', 're', 'collections', 'enum', 'typing',
    'hashlib', 'base64', 'binascii', 'uuid', 'json',
    'html', 'html.entities', 'html.parser', 'urllib.parse',
    'textwrap', 'unicodedata', 'stringprep', 'codecs',
    'dataclasses', 'abc', 'contextlib', 'contextvars',
    'types', 'weakref', 'array', 'bisect', 'heapq',
    'copyreg', 'pickletools',
}


class SecurityError(Exception):
    """Raised when sandbox security policy is violated."""
    pass


class ResourceLimitError(Exception):
    """Raised when resource limits are exceeded."""
    pass


class TimeoutError(Exception):
    """Raised when tool execution exceeds time limit."""
    pass


class RestrictedImport:
    """
    Import hook that blocks dangerous modules and only allows safe ones.
    """
    
    def __init__(self, allowed_modules: Optional[Set[str]] = None):
        self.allowed = allowed_modules or SAFE_MODULES
        self.blocked = DANGEROUS_MODULES
    
    def __call__(self, name: str, globals=None, locals=None, fromlist=(), level=0):
        # Block dangerous modules
        base_module = name.split('.')[0]
        
        if base_module in self.blocked:
            raise SecurityError(f"Import of module '{name}' is not allowed in sandboxed environment")
        
        # Check if full name or any prefix is in allowed list
        # This supports both 'urllib.parse' and 'urllib' when 'urllib.parse' is in SAFE_MODULES
        parts = name.split('.')
        for i in range(len(parts)):
            prefix = '.'.join(parts[:i+1])
            if prefix in self.allowed:
                # Use the stored real __import__ function instead of __builtins__ subscript
                return _real_import(name, globals, locals, fromlist, level)
        
        # If no prefix matches, check if base module is in allowed
        if base_module not in self.allowed:
            raise SecurityError(f"Import of module '{name}' is not permitted. Allowed modules: {sorted(self.allowed)}")
        
        # Use the stored real __import__ function instead of __builtins__ subscript
        return _real_import(name, globals, locals, fromlist, level)


class SafeOpen:
    """
    Restricted file open that only allows access to specific directories.
    """
    
    def __init__(self, allowed_dirs: Optional[List[str]] = None, read_only: bool = True):
        self.allowed_dirs = allowed_dirs or []
        self.read_only = read_only
    
    def _is_path_allowed(self, path: str) -> bool:
        """Check if a path is within allowed directories."""
        # Normalize the path
        try:
            real_path = os.path.realpath(os.path.abspath(path))
        except (OSError, ValueError):
            return False
        
        # Check for path traversal attempts
        if '..' in path or path.startswith('/'):
            # Path traversal detected - reject it
            return False
        
        # Check if path is within allowed directories
        for allowed_dir in self.allowed_dirs:
            try:
                real_allowed = os.path.realpath(os.path.abspath(allowed_dir))
                if real_path.startswith(real_allowed + os.sep) or real_path == real_allowed:
                    return True
            except (OSError, ValueError):
                continue
        
        return False
    
    def __call__(self, file: str, mode: str = 'r', *args, **kwargs):
        if not self._is_path_allowed(file):
            raise SecurityError(f"Access to file '{file}' is not permitted in sandboxed environment")
        
        # In read-only mode, only allow read modes (starting with 'r')
        if self.read_only and not mode.startswith('r'):
            raise SecurityError(f"Write access to file '{file}' is not permitted in sandboxed environment")
        
        return builtins.open(file, mode, *args, **kwargs)


class SafeHttpxClient:
    """
    Restricted HTTP client that enforces network security policies.
    """
    
    # Default allowed domains (can be configured)
    DEFAULT_ALLOWED_DOMAINS: Set[str] = {
        'api.openai.com',
        'api.anthropic.com',
        'api.together.xyz',
        'api.groq.com',
        'api.cohere.com',
        'api.ai21.com',
        'api.perplexity.ai',
        'api.mistral.ai',
        'generativelanguage.googleapis.com',
        'api.replicate.com',
        'api.huggingface.co',
        'api.wolframalpha.com',
        'api.weatherapi.com',  # Weather tool
        'api.frankfurter.app',  # Currency converter
        'en.wikipedia.org',  # Wikipedia
        'export.arxiv.org',  # ArXiv
        'www.youtube.com', 'youtube.com',  # YouTube transcripts
        'api.telegram.org',  # Telegram
    }
    
    # Blocked IP ranges (private networks)
    BLOCKED_IP_PATTERNS: List[str] = [
        r'^127\.',  # Loopback
        r'^10\.',   # Private Class A
        r'^172\.(1[6-9]|2[0-9]|3[0-1])\.',  # Private Class B
        r'^192\.168\.',  # Private Class C
        r'^169\.254\.',  # Link-local
        r'^0\.',    # Current network
        r'^::1$',   # IPv6 loopback
        r'^fc00:',  # IPv6 private
        r'^fe80:',  # IPv6 link-local
    ]
    
    def __init__(self, allowed_domains: Optional[Set[str]] = None, 
                 timeout: float = 30.0,
                 max_response_size: int = 10 * 1024 * 1024):  # 10MB
        self.allowed_domains = allowed_domains or self.DEFAULT_ALLOWED_DOMAINS
        self.timeout = timeout
        self.max_response_size = max_response_size
        self._blocked_patterns = [re.compile(p) for p in self.BLOCKED_IP_PATTERNS]
    
    def _is_domain_allowed(self, url: str) -> bool:
        """Check if a URL's domain is in the allowed list."""
        from urllib.parse import urlparse
        
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            
            # Remove port if present
            if ':' in domain:
                domain = domain.split(':')[0]
            
            # Check exact match or subdomain match
            for allowed in self.allowed_domains:
                if domain == allowed or domain.endswith('.' + allowed):
                    return True
            
            return False
        except Exception:
            return False
    
    def _is_ip_blocked(self, host: str) -> bool:
        """Check if a host/IP is in blocked ranges."""
        for pattern in self._blocked_patterns:
            if pattern.match(host):
                return True
        return False
    
    def request(self, method: str, url: str, **kwargs):
        """Make a restricted HTTP request."""
        import httpx
        import socket
        from urllib.parse import urlparse
        
        # Parse URL to get domain
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        
        # Remove port if present
        if ':' in domain:
            domain = domain.split(':')[0]
        
        # Check domain whitelist
        if not self._is_domain_allowed(url):
            raise SecurityError(f"HTTP requests to '{url}' are not permitted. Domain not in whitelist.")
        
        # Resolve domain and check if it resolves to a blocked IP
        # This prevents attackers from using DNS tricks to access internal services
        try:
            ip = socket.gethostbyname(domain)
            if self._is_ip_blocked(ip):
                raise SecurityError(f"Domain '{domain}' resolves to blocked IP address '{ip}'")
        except socket.gaierror:
            raise SecurityError(f"Could not resolve domain '{domain}' - DNS lookup failed")
        except socket.herror:
            raise SecurityError(f"Could not resolve domain '{domain}' - host not found")
        
        # Enforce timeout
        kwargs['timeout'] = min(kwargs.get('timeout', self.timeout), self.timeout)
        
        # Make the request with size limits
        with httpx.Client() as client:
            response = client.request(method, url, **kwargs)
            
            # Check response size
            content_length = len(response.content)
            if content_length > self.max_response_size:
                raise ResourceLimitError(
                    f"Response size ({content_length} bytes) exceeds maximum allowed ({self.max_response_size} bytes)"
                )
            
            return response
    
    def get(self, url: str, **kwargs):
        return self.request('GET', url, **kwargs)
    
    def post(self, url: str, **kwargs):
        return self.request('POST', url, **kwargs)


def _execute_in_process(code: str, local_vars: Dict[str, Any], result_queue: Queue, 
                        allowed_file_dirs: List[str], read_only_files: bool,
                        allowed_domains: Optional[Set[str]], timeout: float,
                        max_output_size: int, max_memory_mb: int):
    """Execute code in a separate process with restricted environment."""
    try:
        # Enforce memory limit using resource module (Unix only)
        if HAS_RESOURCE_MODULE and max_memory_mb > 0:
            max_bytes = max_memory_mb * 1024 * 1024
            try:
                resource.setrlimit(resource.RLIMIT_AS, (max_bytes, max_bytes))
            except (ValueError, OSError) as e:
                logger.warning(f"Failed to set memory limit: {e}")
        
        # Start tracemalloc for memory monitoring (cross-platform)
        if max_memory_mb > 0:
            tracemalloc.start()
        
        # Rebuild sandbox environment in child process
        sandbox = ToolSandbox(
            timeout=timeout,
            allowed_file_dirs=allowed_file_dirs,
            allowed_domains=allowed_domains,
            read_only_files=read_only_files,
            max_output_size=max_output_size,
            max_memory_mb=max_memory_mb
        )

        result = sandbox._execute_internal(code, local_vars)
        
        # Check memory usage after execution
        if max_memory_mb > 0:
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            peak_mb = peak / (1024 * 1024)
            if peak_mb > max_memory_mb:
                raise ResourceLimitError(
                    f"Memory limit exceeded: {peak_mb:.1f}MB used, limit was {max_memory_mb}MB"
                )
        
        result_queue.put({'success': True, 'result': result.get('result')})

    except SecurityError as e:
        result_queue.put({'success': False, 'error_type': 'SecurityError', 'error_msg': str(e)})
    except ResourceLimitError as e:
        result_queue.put({'success': False, 'error_type': 'ResourceLimitError', 'error_msg': str(e)})
    except TimeoutError as e:
        result_queue.put({'success': False, 'error_type': 'TimeoutError', 'error_msg': str(e)})
    except Exception as e:
        result_queue.put({'success': False, 'error_type': type(e).__name__, 'error_msg': str(e)})



class ToolSandbox:
    """
    Main sandbox class for executing tool code securely.
    """
    
    DEFAULT_TIMEOUT: float = 30.0  # seconds
    DEFAULT_MAX_OUTPUT_SIZE: int = 100 * 1024  # 100KB
    DEFAULT_MAX_MEMORY_MB: int = 100  # MB
    
    def __init__(self, 
                 timeout: Optional[float] = None,
                 max_output_size: Optional[int] = None,
                 allowed_file_dirs: Optional[List[str]] = None,
                 allowed_domains: Optional[Set[str]] = None,
                 read_only_files: bool = True,
                 max_memory_mb: Optional[int] = None):
        self.timeout = timeout or self.DEFAULT_TIMEOUT
        self.max_output_size = max_output_size or self.DEFAULT_MAX_OUTPUT_SIZE
        self.max_memory_mb = max_memory_mb or self.DEFAULT_MAX_MEMORY_MB
        self.allowed_file_dirs = allowed_file_dirs or []
        self.allowed_domains = allowed_domains
        self.read_only_files = read_only_files
        
        # Build restricted globals
        self._globals = self._build_restricted_globals()


    
    def _build_restricted_globals(self) -> Dict[str, Any]:
        """Build a dictionary of restricted globals for sandboxed execution."""
        
        # Start with safe builtins
        safe_builtins = {}
        for name in dir(builtins):
            if name not in DANGEROUS_BUILTINS and not name.startswith('_'):
                safe_builtins[name] = getattr(builtins, name)
        
        # Add our restricted import
        safe_builtins['__import__'] = RestrictedImport()
        
        # Add restricted open if file access is needed
        if self.allowed_file_dirs:
            safe_builtins['open'] = SafeOpen(self.allowed_file_dirs, self.read_only_files)
        
        # Build the globals dict
        restricted_globals = {
            '__builtins__': safe_builtins,
            '__name__': '__sandbox__',
            '__doc__': None,
            '__package__': None,
        }
        
        # Pre-import safe modules
        for module_name in SAFE_MODULES:
            try:
                module = __import__(module_name)
                restricted_globals[module_name] = module
            except ImportError:
                pass
        
        # Add safe httpx client
        restricted_globals['httpx'] = SafeHttpxClient(
            allowed_domains=self.allowed_domains,
            timeout=self.timeout
        )
        
        # Add json module (commonly needed)
        restricted_globals['json'] = json
        
        return restricted_globals
    
    def _check_code_safety(self, code: str) -> None:
        """
        Static analysis to check for dangerous code patterns.
        """
        # Check for dangerous imports
        import_patterns = [
            r'^\s*import\s+(os|sys|subprocess|socket|multiprocessing|ctypes|mmap)',
            r'^\s*from\s+(os|sys|subprocess|socket|multiprocessing|ctypes|mmap)\s+import',
            r'__import__\s*\(',
            r'importlib\s*\.\s*import_module',
        ]
        
        for pattern in import_patterns:
            if re.search(pattern, code, re.MULTILINE | re.IGNORECASE):
                match = re.search(pattern, code, re.MULTILINE | re.IGNORECASE)
                raise SecurityError(f"Dangerous import pattern detected: {match.group()}")
        
        # Check for dangerous builtins
        for dangerous in ['eval(', 'exec(', 'compile(', 'breakpoint(', 'open(']:
            if dangerous in code:
                # Allow open() if it's our SafeOpen (will be checked at runtime)
                if dangerous == 'open(' and self.allowed_file_dirs:
                    continue
                raise SecurityError(f"Dangerous builtin '{dangerous}' detected in code")
        
        # Check for file path traversal
        if '..' in code or '/etc/' in code or 'C:\\\\Windows' in code:
            raise SecurityError("Potential path traversal or system file access detected")
        
        # AST-based analysis to detect dangerous function calls (harder to bypass)
        try:
            import ast
            tree = ast.parse(code)
            dangerous_funcs = {'exec', 'eval', 'compile', 'breakpoint', 'open', '__import__'}
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    # Check direct function calls like exec()
                    if isinstance(node.func, ast.Name) and node.func.id in dangerous_funcs:
                        # Allow open() if it's our SafeOpen (will be checked at runtime)
                        if node.func.id == 'open' and self.allowed_file_dirs:
                            continue
                        raise SecurityError(f"Dangerous call detected: {node.func.id}()")
                    # Check getattr(builtins, "exec")() style calls
                    if isinstance(node.func, ast.Attribute):
                        if node.func.attr in dangerous_funcs:
                            raise SecurityError(f"Dangerous call detected: .{node.func.attr}()")
        except SyntaxError:
            # If code can't be parsed, let it fail at execution time
            pass
    
    def _execute_internal(self, code: str, local_vars: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Internal execution without multiprocessing - used by child process."""
        # Pre-execution safety checks
        self._check_code_safety(code)
        
        # Prepare execution environment
        exec_globals = self._globals.copy()
        exec_locals = local_vars or {}
        
        # Execute
        result_container = {'result': None, 'error': None, 'output': ''}
        
        try:
            exec(code, exec_globals, exec_locals)
            result_container['result'] = exec_locals.get('result')
        except Exception as e:
            result_container['error'] = e
        
        # Check for errors
        if result_container['error']:
            raise result_container['error']
        
        # Check output size
        if result_container['result']:
            result_str = str(result_container['result'])
            if len(result_str) > self.max_output_size:
                raise ResourceLimitError(
                    f"Tool output size ({len(result_str)} chars) exceeds maximum ({self.max_output_size})"
                )
        
        return result_container

    def execute(self, code: str, local_vars: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute code in the sandboxed environment using multiprocessing for true isolation.
        
        Args:
            code: Python code to execute
            local_vars: Variables to inject into the local scope
            
        Returns:
            Dictionary containing execution results and any output variables
            
        Raises:
            SecurityError: If code violates security policy
            ResourceLimitError: If resource limits are exceeded
            TimeoutError: If execution exceeds time limit
        """
        # Pre-execution safety checks (in parent process)
        self._check_code_safety(code)
        
        # Use multiprocessing for true isolation and timeout enforcement
        result_queue = Queue()
        p = Process(
            target=_execute_in_process,
            args=(code, local_vars or {}, result_queue, 
                  self.allowed_file_dirs, self.read_only_files,
                  self.allowed_domains, self.timeout,
                  self.max_output_size, self.max_memory_mb)
        )


        p.start()
        p.join(timeout=self.timeout)
        
        if p.is_alive():
            # Process is still running after timeout - terminate it
            logger.warning(f"Tool execution exceeded {self.timeout} second limit, terminating process")
            p.terminate()
            p.join()
            raise TimeoutError(f"Tool execution exceeded {self.timeout} second limit")
        
        # Check if process exited with error
        if p.exitcode != 0:
            logger.error(f"Tool process exited with code {p.exitcode}")
            raise Exception(f"Tool execution failed with exit code {p.exitcode}")
        
        # Get result from queue
        try:
            result = result_queue.get_nowait()
        except Exception:
            raise Exception("Tool execution failed to return result")
        
        # Check if result contains an error
        if isinstance(result, dict):
            if not result.get('success', False):
                error_type = result.get('error_type', 'Exception')
                error_msg = result.get('error_msg', 'Unknown error')
                
                # Re-raise as appropriate exception type
                if error_type == 'SecurityError':
                    raise SecurityError(error_msg)
                elif error_type == 'ResourceLimitError':
                    raise ResourceLimitError(error_msg)
                elif error_type == 'TimeoutError':
                    raise TimeoutError(error_msg)
                else:
                    raise Exception(f"{error_type}: {error_msg}")
            
            # Return successful result
            return {'result': result.get('result')}
        
        return result




# Convenience function for simple sandboxed execution
def execute_sandboxed(code: str, 
                     args: Dict[str, Any],
                     timeout: float = 30.0,
                     allowed_domains: Optional[Set[str]] = None) -> Any:
    """
    Execute tool code in a sandbox with the given arguments.
    
    Args:
        code: Python code to execute
        args: Arguments to pass to the tool (available as 'args' variable)
        timeout: Maximum execution time in seconds
        allowed_domains: Set of allowed domains for HTTP requests
        
    Returns:
        The 'result' variable from the executed code
    """
    sandbox = ToolSandbox(timeout=timeout, allowed_domains=allowed_domains)
    result = sandbox.execute(code, {'args': args})
    return result['result']
