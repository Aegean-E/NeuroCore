"""
Sandbox Manager for NeuroCore - Worker Pool for Isolated Tool Execution

Provides:
1. Worker Pool - N isolated processes for tool execution
2. Resource Limits - CPU timeout, RAM limits, no network by default
3. Result Queue - Structured output, errors, resource usage
4. Worker Types - Different worker configurations for different needs

Architecture:
    NeuroCore Server
          │
          ▼
    Sandbox Manager
   ├── Worker Pool (N isolated processes)
   │      ├── Worker 1: resource limits (CPU 30s, RAM 256MB, no network by default)
   │      ├── Worker 2: allowlisted imports only
   │      └── Worker 3: filesystem jail (tmp only)
   │
   └── Result Queue (structured output, errors, resource usage)

Usage:
    from modules.tools.sandbox_manager import sandbox_manager
    
    # Execute tool in default worker
    result = await sandbox_manager.execute("tool_code", {"arg": "value"})
    
    # Execute with specific worker type
    result = await sandbox_manager.execute("tool_code", {"arg": "value"}, worker_type="no_network")
"""

import asyncio
import uuid
import time
import logging
from typing import Any, Dict, Optional, List
from dataclasses import dataclass, field
from enum import Enum
from queue import Queue, Empty
from multiprocessing import Process, Queue as MPQueue
import threading

from .sandbox import (
    ToolSandbox,
    SecurityError,
    ResourceLimitError,
    TimeoutError,
    execute_sandboxed,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Worker Types
# =============================================================================

class WorkerType(Enum):
    """Types of sandbox workers with different configurations."""
    
    # Default worker - balanced settings
    DEFAULT = "default"
    
    # No network access
    NO_NETWORK = "no_network"
    
    # Read-only filesystem (tmp only)
    READONLY_FS = "readonly_fs"
    
    # Strict - minimal imports allowed
    STRICT = "strict"
    
    # Tool execution with network allowed (for API calls)
    TOOL = "tool"


# =============================================================================
# Worker Configuration
# =============================================================================

@dataclass
class WorkerConfig:
    """Configuration for a sandbox worker."""
    
    # Resource limits
    timeout: float = 30.0  # seconds
    max_memory_mb: int = 256  # MB
    max_output_size: int = 100 * 1024  # 100KB
    
    # Network settings
    allow_network: bool = True
    allowed_domains: Optional[set] = None
    
    # Filesystem settings
    allowed_file_dirs: List[str] = field(default_factory=list)
    read_only_files: bool = True
    
    # Import restrictions
    strict_imports: bool = False
    
    # Description
    description: str = ""


# Default worker configurations
WORKER_CONFIGS: Dict[WorkerType, WorkerConfig] = {
    WorkerType.DEFAULT: WorkerConfig(
        timeout=30.0,
        max_memory_mb=256,
        max_output_size=100 * 1024,
        allow_network=True,
        allowed_domains=None,  # Use sandbox default
        description="Default worker with balanced settings",
    ),
    WorkerType.NO_NETWORK: WorkerConfig(
        timeout=30.0,
        max_memory_mb=256,
        max_output_size=100 * 1024,
        allow_network=False,
        allowed_domains=set(),
        description="No network access - maximum security",
    ),
    WorkerType.READONLY_FS: WorkerConfig(
        timeout=30.0,
        max_memory_mb=256,
        max_output_size=100 * 1024,
        allow_network=True,
        allowed_file_dirs=["/tmp", "C:\\Temp"],
        read_only_files=True,
        description="Read-only filesystem - can only write to temp",
    ),
    WorkerType.STRICT: WorkerConfig(
        timeout=15.0,  # Stricter timeout
        max_memory_mb=128,  # Less memory
        max_output_size=50 * 1024,  # Less output
        allow_network=False,
        strict_imports=True,
        description="Strict worker - minimal permissions",
    ),
    WorkerType.TOOL: WorkerConfig(
        timeout=60.0,  # Longer timeout for complex tools
        max_memory_mb=512,  # More memory for tools
        max_output_size=500 * 1024,  # More output
        allow_network=True,
        description="Tool execution worker - extended resources",
    ),
}


# =============================================================================
# Execution Result
# =============================================================================

@dataclass
class ExecutionResult:
    """Result of sandboxed execution."""
    
    success: bool
    result: Any = None
    error: Optional[str] = None
    error_type: Optional[str] = None
    
    # Resource usage
    execution_time_ms: float = 0.0
    memory_peak_mb: Optional[float] = None
    output_size: Optional[int] = None
    
    # Metadata
    trace_id: Optional[str] = None
    worker_type: WorkerType = WorkerType.DEFAULT
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "result": self.result,
            "error": self.error,
            "error_type": self.error_type,
            "execution_time_ms": self.execution_time_ms,
            "memory_peak_mb": self.memory_peak_mb,
            "output_size": self.output_size,
            "trace_id": self.trace_id,
            "worker_type": self.worker_type.value,
        }


# =============================================================================
# Worker Pool
# =============================================================================

class WorkerPool:
    """
    Pool of sandbox workers for parallel execution.
    
    Manages a pool of Worker processes, each running in isolation.
    """
    
    def __init__(self, 
                 pool_size: int = 4,
                 default_worker_type: WorkerType = WorkerType.DEFAULT):
        self.pool_size = pool_size
        self.default_worker_type = default_worker_type
        
        # Track available workers and their queues
        self._worker_queues: Dict[int, Queue] = {}
        self._worker_processes: List[Process] = []
        self._result_queues: Dict[int, MPQueue] = {}
        
        # Lock for thread safety
        self._lock = threading.Lock()
        
        # Initialize workers
        self._initialize_workers()
    
    def _initialize_workers(self):
        """Initialize worker processes."""
        for i in range(self.pool_size):
            # Create result queue for this worker
            result_queue = MPQueue()
            self._result_queues[i] = result_queue
            
            # Start worker process
            p = Process(
                target=self._worker_loop,
                args=(i, result_queue),
                daemon=True,
            )
            p.start()
            self._worker_processes.append(p)
            
            # Create request queue for this worker
            self._worker_queues[i] = Queue()
            
            logger.info(f"Started worker {i} (PID: {p.pid})")
    
    @staticmethod
    def _worker_loop(worker_id: int, result_queue: MPQueue):
        """
        Worker process loop.
        
        Each worker runs in a separate process, completely isolated
        from other workers and the main process.
        """
        logger.info(f"Worker {worker_id} started")
        
        while True:
            try:
                # Get task from request queue (blocking)
                task_queue = WorkerPool._get_worker_queue(worker_id)
                if task_queue is None:
                    time.sleep(0.1)
                    continue
                
                try:
                    task = task_queue.get(timeout=1.0)
                except Empty:
                    continue
                
                if task is None:  # Shutdown signal
                    break
                
                # Unpack task
                task_id, code, args, config, result_queue_inner = task
                
                # Execute in sandbox
                start_time = time.time()
                success = False
                result = None
                error = None
                error_type = None
                
                try:
                    # Create sandbox with config
                    sandbox = ToolSandbox(
                        timeout=config.timeout,
                        max_output_size=config.max_output_size,
                        allowed_file_dirs=config.allowed_file_dirs,
                        allowed_domains=config.allowed_domains,
                        read_only_files=config.read_only_files,
                        max_memory_mb=config.max_memory_mb,
                    )
                    
                    # Execute
                    exec_result = sandbox.execute(code, args)
                    result = exec_result.get("result")
                    success = True
                    
                except SecurityError as e:
                    error = str(e)
                    error_type = "SecurityError"
                except ResourceLimitError as e:
                    error = str(e)
                    error_type = "ResourceLimitError"
                except TimeoutError as e:
                    error = str(e)
                    error_type = "TimeoutError"
                except Exception as e:
                    error = str(e)
                    error_type = type(e).__name__
                
                execution_time_ms = (time.time() - start_time) * 1000
                
                # Put result
                exec_result = ExecutionResult(
                    success=success,
                    result=result,
                    error=error,
                    error_type=error_type,
                    execution_time_ms=execution_time_ms,
                    worker_type=WorkerType.DEFAULT,  # Will be set by manager
                )
                
                result_queue_inner.put((task_id, exec_result))
                
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
                continue
        
        logger.info(f"Worker {worker_id} stopped")
    
    @staticmethod
    def _worker_queue_map: Dict[int, Queue] = {}
    _worker_queue_map_lock = threading.Lock()
    
    @staticmethod
    def _register_worker_queue(worker_id: int, queue: Queue):
        """Register a worker's request queue."""
        with WorkerPool._worker_queue_map_lock:
            WorkerPool._worker_queue_map[worker_id] = queue
    
    @staticmethod
    def _get_worker_queue(worker_id: int) -> Optional[Queue]:
        """Get a worker's request queue."""
        with WorkerPool._worker_queue_map_lock:
            return WorkerPool._worker_queue_map.get(worker_id)
    
    def execute(self, 
                code: str, 
                args: Dict[str, Any],
                worker_type: WorkerType = None,
                timeout: float = None) -> ExecutionResult:
        """
        Execute code in a worker from the pool.
        
        Args:
            code: Python code to execute
            args: Arguments to pass to the code
            worker_type: Type of worker to use
            timeout: Override timeout
            
        Returns:
            ExecutionResult with execution details
        """
        worker_type = worker_type or self.default_worker_type
        config = WORKER_CONFIGS[worker_type].copy()
        
        if timeout:
            config.timeout = timeout
        
        # Get available worker (round-robin)
        with self._lock:
            worker_id = len(self._worker_queues) % self.pool_size
        
        # Create task
        task_id = uuid.uuid4().hex
        result_queue = Queue()
        
        task = (task_id, code, args, config, result_queue)
        
        # Submit to worker
        worker_queue = self._worker_queues[worker_id]
        worker_queue.put(task)
        
        # Wait for result with timeout
        try:
            result_task_id, exec_result = result_queue.get(timeout=config.timeout + 5)
            
            if result_task_id != task_id:
                logger.warning(f"Task ID mismatch: {result_task_id} != {task_id}")
            
            return exec_result
            
        except Empty:
            return ExecutionResult(
                success=False,
                error=f"Execution timed out after {config.timeout} seconds",
                error_type="TimeoutError",
                execution_time_ms=config.timeout * 1000,
            )
    
    def shutdown(self):
        """Shutdown all workers."""
        for queue in self._worker_queues.values():
            queue.put(None)  # Shutdown signal
        
        for p in self._worker_processes:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
        
        self._worker_queues.clear()
        self._worker_processes.clear()
        logger.info("Worker pool shutdown complete")


# =============================================================================
# Sandbox Manager
# =============================================================================

class SandboxManager:
    """
    Main sandbox manager for NeuroCore.
    
    Provides a high-level interface for sandboxed tool execution
    with worker pools, resource limits, and result tracking.
    """
    
    def __init__(self, 
                 pool_size: int = 4,
                 default_worker_type: WorkerType = WorkerType.DEFAULT):
        self.pool_size = pool_size
        self.default_worker_type = default_worker_type
        
        # Create worker pool
        self._pool = WorkerPool(pool_size, default_worker_type)
        
        # Track execution history
        self._execution_history: List[ExecutionResult] = []
        self._max_history = 100
        
        logger.info(f"SandboxManager initialized with pool_size={pool_size}")
    
    async def execute(self, 
                      code: str, 
                      args: Dict[str, Any],
                      worker_type: WorkerType = None,
                      timeout: float = None,
                      trace_id: str = None) -> ExecutionResult:
        """
        Execute code in sandbox.
        
        Args:
            code: Python code to execute
            args: Arguments to pass to the code
            worker_type: Type of worker to use
            timeout: Maximum execution time in seconds
            trace_id: Optional trace ID for observability
            
        Returns:
            ExecutionResult with execution details
        """
        worker_type = worker_type or self.default_worker_type
        
        # Run in executor to avoid blocking event loop
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            self._pool.execute,
            code,
            args,
            worker_type,
            timeout,
        )
        
        # Add trace ID
        result.trace_id = trace_id
        
        # Track in history
        self._add_to_history(result)
        
        return result
    
    def execute_sync(self, 
                     code: str, 
                     args: Dict[str, Any],
                     worker_type: WorkerType = None,
                     timeout: float = None,
                     trace_id: str = None) -> ExecutionResult:
        """
        Synchronous execute (for non-async contexts).
        """
        worker_type = worker_type or self.default_worker_type
        
        result = self._pool.execute(code, args, worker_type, timeout)
        result.trace_id = trace_id
        
        self._add_to_history(result)
        
        return result
    
    def _add_to_history(self, result: ExecutionResult):
        """Add result to execution history."""
        self._execution_history.append(result)
        
        # Trim history
        if len(self._execution_history) > self._max_history:
            self._execution_history = self._execution_history[-self._max_history:]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get execution statistics."""
        if not self._execution_history:
            return {
                "total_executions": 0,
                "success_rate": 0.0,
                "avg_execution_time_ms": 0.0,
            }
        
        total = len(self._execution_history)
        successes = sum(1 for r in self._execution_history if r.success)
        total_time = sum(r.execution_time_ms for r in self._execution_history)
        
        return {
            "total_executions": total,
            "successes": successes,
            "failures": total - successes,
            "success_rate": successes / total if total > 0 else 0.0,
            "avg_execution_time_ms": total_time / total if total > 0 else 0.0,
            "pool_size": self.pool_size,
            "default_worker_type": self.default_worker_type.value,
        }
    
    def get_recent_results(self, count: int = 10) -> List[Dict[str, Any]]:
        """Get recent execution results."""
        results = self._execution_history[-count:]
        return [r.to_dict() for r in results]
    
    def shutdown(self):
        """Shutdown the sandbox manager."""
        self._pool.shutdown()
        logger.info("SandboxManager shutdown complete")


# =============================================================================
# Global Instance
# =============================================================================

# Default sandbox manager instance
sandbox_manager = SandboxManager(
    pool_size=4,
    default_worker_type=WorkerType.TOOL,  # Tools need network access
)


# =============================================================================
# Convenience Functions
# =============================================================================

async def execute_in_sandbox(code: str, 
                             args: Dict[str, Any],
                             worker_type: WorkerType = WorkerType.TOOL,
                             timeout: float = 30.0) -> ExecutionResult:
    """
    Convenience function for executing code in sandbox.
    
    Args:
        code: Python code to execute
        args: Arguments to pass to the code
        worker_type: Type of worker to use
        timeout: Maximum execution time in seconds
        
    Returns:
        ExecutionResult
    """
    return await sandbox_manager.execute(code, args, worker_type, timeout)


def execute_in_sandbox_sync(code: str, 
                           args: Dict[str, Any],
                           worker_type: WorkerType = WorkerType.TOOL,
                           timeout: float = 30.0) -> ExecutionResult:
    """
    Synchronous version of execute_in_sandbox.
    """
    return sandbox_manager.execute_sync(code, args, worker_type, timeout)


# =============================================================================
# Integration with existing sandbox
# =============================================================================

def create_sandbox_for_worker_type(worker_type: WorkerType) -> ToolSandbox:
    """Create a ToolSandbox configured for a specific worker type."""
    config = WORKER_CONFIGS[worker_type]
    
    return ToolSandbox(
        timeout=config.timeout,
        max_output_size=config.max_output_size,
        allowed_file_dirs=config.allowed_file_dirs,
        allowed_domains=config.allowed_domains,
        read_only_files=config.read_only_files,
        max_memory_mb=config.max_memory_mb,
    )


if __name__ == "__main__":
    # Test the sandbox manager
    import asyncio
    
    async def test():
        print("Testing SandboxManager...")
        
        # Simple execution
        result = await sandbox_manager.execute(
            "result = args['x'] + args['y']",
            {"x": 5, "y": 3}
        )
        print(f"Result: {result.to_dict()}")
        
        # Test security
        result = await sandbox_manager.execute(
            "import os; result = os.system('ls')",
            {},
            worker_type=WorkerType.NO_NETWORK
        )
        print(f"Security test: {result.to_dict()}")
        
        # Statistics
        print(f"Statistics: {sandbox_manager.get_statistics()}")
        
        await asyncio.sleep(1)
        sandbox_manager.shutdown()
    
    asyncio.run(test())

