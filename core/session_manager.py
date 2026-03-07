"""
Session Manager for NeuroCore

Provides:
1. Session persistence across restarts (session.json)
2. Append-only execution trace (execution_trace.jsonl)
3. State management for agent sessions

Usage:
    from core.session_manager import session_manager, get_session_manager
    
    # Get or create session
    session_id = session_manager.load_or_create_session()
    
    # Log tool events
    session_manager.log_tool_call("calculator", {"expression": "2+2"})
    session_manager.log_tool_result("calculator", "4", duration_ms=5.2)
    
    # Log LLM calls
    session_manager.log_llm_call("gpt-4", tokens=150)
    
    # Save session state
    session_manager.update_state({"goal": "analyze data", "step": 3})
    session_manager.save_state()
    
    # Get trace for replay/debugging
    trace = session_manager.get_trace()
"""

import json
import os
import uuid
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, List
from contextlib import contextmanager

# Configuration
SESSION_FILE = "data/session.json"
TRACE_FILE = "data/execution_trace.jsonl"
AGENT_ID = "neurocore"


class TraceWriter:
    """
    Append-only JSON Lines trace writer.
    Thread-safe.
    """
    
    def __init__(self, trace_file: str = TRACE_FILE):
        self.trace_file = trace_file
        self._lock = threading.Lock()
        self._ensure_directory()
    
    def _ensure_directory(self):
        """Ensure trace file directory exists."""
        directory = os.path.dirname(self.trace_file)
        if directory:
            os.makedirs(directory, exist_ok=True)
    
    def append(self, event: Dict[str, Any]) -> None:
        """
        Append a JSON event to the trace file.
        Thread-safe operation.
        """
        with self._lock:
            try:
                # Ensure directory exists
                directory = os.path.dirname(self.trace_file)
                if directory:
                    os.makedirs(directory, exist_ok=True)
                
                # Format as JSON Line
                line = json.dumps(event, default=str) + '\n'
                
                # Open in append mode and write
                with open(self.trace_file, 'a') as f:
                    f.write(line)
                    
            except Exception as e:
                raise e
    
    def read_all(self) -> List[Dict[str, Any]]:
        """Read all trace events from file."""
        with self._lock:
            if not os.path.exists(self.trace_file):
                return []
            
            events = []
            with open(self.trace_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            events.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
            return events
    
    def read_since(self, timestamp: float) -> List[Dict[str, Any]]:
        """Read trace events since given timestamp."""
        all_events = self.read_all()
        return [e for e in all_events if self._parse_timestamp(e.get('timestamp', '')) > timestamp]
    
    def _parse_timestamp(self, ts: str) -> float:
        """Parse ISO timestamp to unix timestamp."""
        try:
            dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
            return dt.timestamp()
        except (ValueError, AttributeError):
            return 0.0
    
    def clear(self) -> None:
        """Clear trace file (for testing)."""
        with self._lock:
            if os.path.exists(self.trace_file):
                os.remove(self.trace_file)


class SessionManager:
    """
    Manages agent session persistence and tracing.
    
    Provides:
    - Persistent session_id across restarts
    - Session state (goals, plan, etc.)
    - Append-only execution trace
    """
    
    def __init__(
        self,
        session_file: str = SESSION_FILE,
        trace_file: str = TRACE_FILE
    ):
        self.session_file = session_file
        self.trace_writer = TraceWriter(trace_file)
        
        self._sync_lock = threading.Lock()
        self._session_id: Optional[str] = None
        self._state: Dict[str, Any] = {}
        self._tick: int = 0
        
        # Ensure data directory exists
        self._ensure_directory()
        
        # Load existing session or create new one
        self._load_session()
    
    def _ensure_directory(self):
        """Ensure session file directory exists."""
        directory = os.path.dirname(self.session_file)
        if directory:
            os.makedirs(directory, exist_ok=True)
    
    def _load_session(self) -> None:
        """Load existing session or create new one."""
        with self._sync_lock:
            if os.path.exists(self.session_file):
                try:
                    with open(self.session_file, 'r') as f:
                        data = json.load(f)
                        self._session_id = data.get('session_id')
                        self._state = data.get('state', {})
                        self._tick = data.get('tick', 0)
                except (json.JSONDecodeError, IOError):
                    # Invalid session file, create new
                    self._create_new_session()
            else:
                self._create_new_session()
    
    def _create_new_session(self) -> None:
        """Create a new session."""
        self._session_id = f"sess-{uuid.uuid4().hex[:8]}"
        self._state = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "agent_id": AGENT_ID,
        }
        self._tick = 0
        self._save_session_unlocked()
    
    def _save_session_unlocked(self) -> None:
        """Save session (must be called with lock held)."""
        try:
            data = {
                "session_id": self._session_id,
                "state": self._state,
                "tick": self._tick,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            
            # Ensure directory
            directory = os.path.dirname(self.session_file)
            if directory:
                os.makedirs(directory, exist_ok=True)
            
            # Atomic write
            temp_file = self.session_file + ".tmp"
            with open(temp_file, 'w') as f:
                json.dump(data, f, indent=2)
            os.replace(temp_file, self.session_file)
        except Exception as e:
            temp_file = self.session_file + ".tmp"
            if os.path.exists(temp_file):
                os.remove(temp_file)
            raise e
    
    def load_or_create_session(self) -> str:
        """
        Get current session_id, creating one if needed.
        This is the main entry point for session management.
        """
        with self._sync_lock:
            if not self._session_id:
                self._load_session()
            return self._session_id
    
    def get_session_id(self) -> Optional[str]:
        """Get current session_id."""
        with self._sync_lock:
            return self._session_id
    
    def get_state(self) -> Dict[str, Any]:
        """Get current session state."""
        with self._sync_lock:
            return self._state.copy()
    
    def update_state(self, updates: Dict[str, Any]) -> None:
        """Update session state."""
        with self._sync_lock:
            self._state.update(updates)
    
    def save_state(self) -> None:
        """Persist session state to disk."""
        with self._sync_lock:
            self._save_session_unlocked()
    
    def increment_tick(self) -> int:
        """Increment and return current tick."""
        with self._sync_lock:
            self._tick += 1
            return self._tick
    
    def get_tick(self) -> int:
        """Get current tick."""
        with self._sync_lock:
            return self._tick
    
    def reset_session(self) -> str:
        """Create a new session (for testing or new conversation)."""
        with self._sync_lock:
            self._create_new_session()
            return self._session_id
    
    # =========================================================================
    # Tracing Methods
    # =========================================================================
    
    def _make_timestamp(self) -> str:
        """Generate ISO timestamp."""
        return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
    
    def log_tool_call(
        self,
        tool: str,
        input_data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Log a tool call event.
        
        Args:
            tool: Tool name
            input_data: Tool input arguments
        """
        tick = self.increment_tick()
        event = {
            "timestamp": self._make_timestamp(),
            "session_id": self._session_id,
            "tick": tick,
            "event": "tool_call",
            "tool": tool,
            "input": input_data,
        }
        self.trace_writer.append(event)
    
    def log_tool_result(
        self,
        tool: str,
        output: Any,
        success: bool = True,
        duration_ms: Optional[float] = None,
        error: Optional[str] = None,
    ) -> None:
        """
        Log a tool result event.
        
        Args:
            tool: Tool name
            output: Tool output/result
            success: Whether execution succeeded
            duration_ms: Execution time in milliseconds
            error: Error message if failed
        """
        event = {
            "timestamp": self._make_timestamp(),
            "session_id": self._session_id,
            "tick": self._tick,
            "event": "tool_result",
            "tool": tool,
            "output": str(output) if output is not None else None,
            "success": success,
        }
        
        if duration_ms is not None:
            event["duration_ms"] = round(duration_ms, 2)
        
        if error:
            event["error"] = error
        
        self.trace_writer.append(event)
    
    def log_llm_call(
        self,
        model: str,
        tokens: Optional[int] = None,
        latency_ms: Optional[float] = None,
    ) -> None:
        """
        Log an LLM call event.
        
        Args:
            model: Model name
            tokens: Token count (if available)
            latency_ms: Latency in milliseconds
        """
        tick = self.increment_tick()
        event = {
            "timestamp": self._make_timestamp(),
            "session_id": self._session_id,
            "tick": tick,
            "event": "llm_call",
            "model": model,
        }
        
        if tokens is not None:
            event["tokens"] = tokens
        
        if latency_ms is not None:
            event["latency_ms"] = round(latency_ms, 2)
        
        self.trace_writer.append(event)
    
    def log_agent_event(
        self,
        event_type: str,
        data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Log a general agent event.
        
        Args:
            event_type: Type of event (e.g., "agent_start", "agent_end", "replan")
            data: Additional event data
        """
        tick = self.increment_tick()
        event = {
            "timestamp": self._make_timestamp(),
            "session_id": self._session_id,
            "tick": tick,
            "event": event_type,
        }
        
        if data:
            event.update(data)
        
        self.trace_writer.append(event)
    
    def log_rlm_event(
        self,
        event_type: str,
        data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Log an RLM-specific event.
        
        Args:
            event_type: Type of RLM event (e.g., "sub_call", "set_final", "stdout")
            data: Additional event data
        """
        tick = self.increment_tick()
        event = {
            "timestamp": self._make_timestamp(),
            "session_id": self._session_id,
            "tick": tick,
            "event": f"rlm_{event_type}",
        }
        
        if data:
            event.update(data)
        
        self.trace_writer.append(event)
    
    def get_trace(self) -> List[Dict[str, Any]]:
        """Get all trace events."""
        return self.trace_writer.read_all()
    
    def get_trace_since(self, timestamp: float) -> List[Dict[str, Any]]:
        """Get trace events since given timestamp."""
        return self.trace_writer.read_since(timestamp)
    
    def get_trace_summary(self, limit: int = 5, since: Optional[float] = None) -> Dict[str, Any]:
        """
        Get a summary of trace events.
        
        Args:
            limit: Number of recent events to include (default: 5)
            since: Unix timestamp to filter events (optional). If provided, only events after this time are included.
            
        Returns:
            Dict with:
                - session_id: Current session ID
                - total_llm_calls: Count of llm_call events
                - total_tool_calls: Count of tool_call events
                - total_tool_failures: Count of tool_result events where success=False
                - last_events: List of recent events (up to limit)
        """
        # Get trace events (filtered by time if 'since' is provided)
        if since is not None:
            trace = self.get_trace_since(since)
        else:
            trace = self.get_trace()
        
        total_llm_calls = sum(1 for e in trace if e.get("event") == "llm_call")
        total_tool_calls = sum(1 for e in trace if e.get("event") == "tool_call")
        total_tool_failures = sum(
            1 for e in trace 
            if e.get("event") == "tool_result" and e.get("success") is False
        )
        
        # Get last N events
        last_events = trace[-limit:] if limit > 0 else []
        
        return {
            "session_id": self._session_id,
            "total_llm_calls": total_llm_calls,
            "total_tool_calls": total_tool_calls,
            "total_tool_failures": total_tool_failures,
            "last_events": last_events,
        }
    
    @contextmanager
    def trace_context(self, operation: str):
        """
        Context manager for tracing operations with timing.
        
        Usage:
            with session_manager.trace_context("my_operation"):
                # do work
                pass
        """
        start_time = time.time()
        self.log_agent_event(f"{operation}_start")
        
        try:
            yield
        finally:
            duration_ms = (time.time() - start_time) * 1000
            self.log_agent_event(
                f"{operation}_end",
                {"duration_ms": round(duration_ms, 2)}
            )


# Global singleton instance
_session_manager: Optional['SessionManager'] = None
_init_lock = threading.Lock()


def get_session_manager() -> 'SessionManager':
    """Get or create the global SessionManager instance."""
    global _session_manager
    
    if _session_manager is None:
        with _init_lock:
            if _session_manager is None:
                _session_manager = SessionManager()
    
    return _session_manager


# Convenience singleton - gets initialized on first import
session_manager = get_session_manager()

