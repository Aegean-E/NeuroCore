"""
Observability Infrastructure for NeuroCore

Provides:
1. Distributed Tracing - trace_id propagation through all operations
2. Metrics Collection - latency, token usage, tool success rates
3. Structured Logging - JSON formatted, searchable logs

Usage:
    from core.observability import trace_context, metrics, structured_logger
    
    # Start a trace
    with trace_context("flow_execution") as ctx:
        ctx.add_span("node_execution", node_id="chat_input")
        # ... do work ...
        metrics.increment("node.executed", node_type="chat")
        metrics.timing("node.latency", elapsed_ms)
"""

import uuid
import time
import json
import logging
import threading
import contextvars
import contextlib
from typing import Optional, Dict, Any, List, Callable
from datetime import datetime
from functools import wraps
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from enum import Enum

# Thread-local trace context
_trace_ctx: contextvars.ContextVar[Optional['TraceContext']] = contextvars.ContextVar('trace_ctx', default=None)


# =============================================================================
# Distributed Tracing
# =============================================================================

class SpanKind(Enum):
    """Types of spans in distributed tracing."""
    INTERNAL = "internal"
    SERVER = "server"
    CLIENT = "client"
    PRODUCER = "producer"
    CONSUMER = "consumer"


@dataclass
class Span:
    """Represents a single operation in a trace."""
    name: str
    span_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    parent_id: Optional[str] = None
    trace_id: Optional[str] = None
    kind: SpanKind = SpanKind.INTERNAL
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    status: str = "ok"  # ok, error
    error_message: Optional[str] = None
    
    def finish(self, status: str = "ok", error: Optional[str] = None):
        """Mark span as finished."""
        self.end_time = time.time()
        self.status = status
        self.error_message = error
    
    @property
    def duration_ms(self) -> float:
        """Get duration in milliseconds."""
        if self.end_time is None:
            return (time.time() - self.start_time) * 1000
        return (self.end_time - self.start_time) * 1000
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "trace_id": self.trace_id,
            "name": self.name,
            "kind": self.kind.value,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "attributes": self.attributes,
            "status": self.status,
            "error_message": self.error_message,
        }


@dataclass 
class TraceContext:
    """Context for distributed tracing."""
    trace_id: str
    spans: List[Span] = field(default_factory=list)
    current_span: Optional[Span] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.trace_id:
            self.trace_id = uuid.uuid4().hex
    
    def add_span(self, name: str, kind: SpanKind = SpanKind.INTERNAL, 
                 attributes: Optional[Dict[str, Any]] = None, 
                 parent_id: Optional[str] = None) -> Span:
        """Create and start a new span."""
        span = Span(
            name=name,
            trace_id=self.trace_id,
            kind=kind,
            parent_id=parent_id or (self.current_span.span_id if self.current_span else None),
            attributes=attributes or {},
        )
        self.spans.append(span)
        self.current_span = span
        return span
    
    def end_span(self, status: str = "ok", error: Optional[str] = None):
        """End the current span."""
        if self.current_span:
            self.current_span.finish(status=status, error=error)
            self.current_span = None
    
    def get_trace(self) -> List[Dict[str, Any]]:
        """Get complete trace as list of span dicts."""
        return [span.to_dict() for span in self.spans]
    
    def __enter__(self):
        _trace_ctx.set(self)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        _trace_ctx.set(None)
        if exc_type:
            if self.current_span:
                self.current_span.finish(status="error", error=str(exc_val))
        return False


# =============================================================================
# Trace Context Management
# =============================================================================

def get_trace_context() -> Optional[TraceContext]:
    """Get current trace context."""
    return _trace_ctx.get()


def get_trace_id() -> Optional[str]:
    """Get current trace ID."""
    ctx = _trace_ctx.get()
    return ctx.trace_id if ctx else None


def get_or_create_trace_context() -> TraceContext:
    """Get current trace context or create a new one."""
    ctx = _trace_ctx.get()
    if ctx is None:
        ctx = TraceContext(trace_id=uuid.uuid4().hex)
        _trace_ctx.set(ctx)
    return ctx


def create_trace_context(trace_id: Optional[str] = None) -> TraceContext:
    """Create a new trace context with optional trace ID."""
    ctx = TraceContext(trace_id=trace_id or uuid.uuid4().hex)
    _trace_ctx.set(ctx)
    return ctx


@contextlib.contextmanager
def trace(span_name: str, kind: SpanKind = SpanKind.INTERNAL, 
           attributes: Optional[Dict[str, Any]] = None):
    """
    Context manager for creating spans.
    
    Usage:
        with trace("process_request") as ctx:
            ctx.add_span("database_query", attributes={"query": "SELECT *"})
    """
    ctx = get_or_create_trace_context()
    span = ctx.add_span(span_name, kind=kind, attributes=attributes)
    try:
        yield ctx
    except Exception as e:
        span.finish(status="error", error=str(e))
        raise
    finally:
        ctx.end_span()


# Thread-local trace context for async operations
_trace_ctx_async: contextvars.ContextVar[Optional['TraceContext']] = contextvars.ContextVar('trace_ctx_async', default=None)


@contextlib.contextmanager
def trace_async(span_name: str, kind: SpanKind = SpanKind.INTERNAL, 
                attributes: Optional[Dict[str, Any]] = None):
    """
    Context manager for creating spans in async context.
    Creates a new TraceContext per async call to avoid concurrent span clobbering.
    
    Usage:
        with trace_async("process_request") as ctx:
            ctx.add_span("database_query", attributes={"query": "SELECT *"})
    """
    # Create a new TraceContext for this async operation
    ctx = TraceContext(trace_id=uuid.uuid4().hex)
    _trace_ctx_async.set(ctx)
    span = ctx.add_span(span_name, kind=kind, attributes=attributes)
    try:
        yield ctx
    except Exception as e:
        span.finish(status="error", error=str(e))
        raise
    finally:
        ctx.end_span()
        _trace_ctx_async.set(None)


def traced(span_name: str = None, kind: SpanKind = SpanKind.INTERNAL):
    """
    Decorator for automatic tracing of functions.
    
    Usage:
        @traced("my_function")
        def my_function():
            ...
    """
    def decorator(func: Callable) -> Callable:
        name = span_name or func.__name__
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            with trace(name, kind=kind) as ctx:
                ctx.current_span.attributes["function"] = func.__name__
                return func(*args, **kwargs)
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Use trace_async to create a new TraceContext for each async call
            # to avoid concurrent coroutines clobbering each other's current_span
            with trace_async(name, kind=kind) as ctx:
                ctx.current_span.attributes["function"] = func.__name__
                return await func(*args, **kwargs)
        
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


# =============================================================================
# Metrics Collection
# =============================================================================

class Metrics:
    """Simple metrics collector with counters, timers, and gauges."""
    
    def __init__(self):
        self._counters: Dict[str, float] = defaultdict(float)
        self._timings: Dict[str, List[float]] = defaultdict(list)
        self._gauges: Dict[str, float] = {}
        self._lock = threading.RLock()
        self._max_timings_per_metric = 1000  # Keep last 1000 timings
    
    def increment(self, metric: str, value: float = 1.0, tags: Optional[Dict[str, str]] = None):
        """Increment a counter metric."""
        key = self._make_key(metric, tags)
        with self._lock:
            self._counters[key] += value
    
    def timing(self, metric: str, duration_ms: float, tags: Optional[Dict[str, str]] = None):
        """Record a timing metric in milliseconds."""
        key = self._make_key(metric, tags)
        with self._lock:
            self._timings[key].append(duration_ms)
            # Keep only last N timings
            if len(self._timings[key]) > self._max_timings_per_metric:
                self._timings[key] = self._timings[key][-self._max_timings_per_metric:]
    
    def gauge(self, metric: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Set a gauge value."""
        key = self._make_key(metric, tags)
        with self._lock:
            self._gauges[key] = value
    
    def _make_key(self, metric: str, tags: Optional[Dict[str, str]] = None) -> str:
        """Create metric key with tags."""
        if not tags:
            return metric
        tag_str = ",".join(f"{k}={v}" for k, v in sorted(tags.items()))
        return f"{metric}{{{tag_str}}}"
    
    def get_counter(self, metric: str, tags: Optional[Dict[str, str]] = None) -> float:
        """Get counter value."""
        key = self._make_key(metric, tags)
        with self._lock:
            return self._counters.get(key, 0.0)
    
    def get_timing_stats(self, metric: str, tags: Optional[Dict[str, str]] = None) -> Dict[str, float]:
        """Get timing statistics (min, max, avg, p50, p95, p99)."""
        key = self._make_key(metric, tags)
        with self._lock:
            timings = self._timings.get(key, [])
            if not timings:
                return {"count": 0, "min": 0, "max": 0, "avg": 0, "p50": 0, "p95": 0, "p99": 0}
            
            sorted_timings = sorted(timings)
            n = len(sorted_timings)
            
            def percentile(p: float) -> float:
                idx = int(n * p)
                return sorted_timings[min(idx, n - 1)]
            
            return {
                "count": n,
                "min": sorted_timings[0],
                "max": sorted_timings[-1],
                "avg": sum(timings) / n,
                "p50": percentile(0.5),
                "p95": percentile(0.95),
                "p99": percentile(0.99),
            }
    
    def get_gauge(self, metric: str, tags: Optional[Dict[str, str]] = None) -> Optional[float]:
        """Get gauge value."""
        key = self._make_key(metric, tags)
        with self._lock:
            return self._gauges.get(key)
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all metrics."""
        with self._lock:
            result = {
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
                "timings": {},
            }
            for key, timings in self._timings.items():
                if timings:
                    sorted_timings = sorted(timings)
                    n = len(sorted_timings)
                    
                    def percentile(p: float) -> float:
                        idx = int(n * p)
                        return sorted_timings[min(idx, n - 1)]
                    
                    result["timings"][key] = {
                        "count": n,
                        "min": sorted_timings[0],
                        "max": sorted_timings[-1],
                        "avg": sum(timings) / n,
                        "p50": percentile(0.5),
                        "p95": percentile(0.95),
                    }
            return result
    
    def reset(self):
        """Reset all metrics."""
        with self._lock:
            self._counters.clear()
            self._timings.clear()
            self._gauges.clear()


# Timing context manager
@contextlib.contextmanager
def timed(metric: str, tags: Optional[Dict[str, str]] = None):
    """Context manager for timing operations."""
    start = time.time()
    try:
        yield
    finally:
        duration_ms = (time.time() - start) * 1000
        metrics.timing(metric, duration_ms, tags)


def timed_decorator(metric: str, tags: Optional[Dict[str, str]] = None):
    """Decorator for timing functions."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start = time.time()
            try:
                return func(*args, **kwargs)
            finally:
                duration_ms = (time.time() - start) * 1000
                metrics.timing(metric, duration_ms, tags)
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start = time.time()
            try:
                return await func(*args, **kwargs)
            finally:
                duration_ms = (time.time() - start) * 1000
                metrics.timing(metric, duration_ms, tags)
        
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


# Global metrics instance
metrics = Metrics()


# =============================================================================
# Structured Logging
# =============================================================================

class StructuredLogger:
    """
    Structured JSON logger for NeuroCore.
    
    Replaces print() statements with searchable, filterable JSON logs.
    """
    
    def __init__(self, name: str = "neurocore"):
        self.logger = logging.getLogger(name)
        self._ensure_handler()
    
    def _ensure_handler(self):
        """Ensure logger has a handler configured."""
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(JsonFormatter())
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def _log(self, level: int, event: str, message: str = "", 
             trace_id: Optional[str] = None, **kwargs):
        """Internal log method."""
        # Get trace context if available
        if trace_id is None:
            ctx = get_trace_context()
            if ctx:
                trace_id = ctx.trace_id
        
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "event": event,
            "message": message,
            "trace_id": trace_id,
        }
        
        # Add current span info if available
        ctx = get_trace_context()
        if ctx and ctx.current_span:
            log_data["span"] = {
                "name": ctx.current_span.name,
                "span_id": ctx.current_span.span_id,
            }
        
        # Add any extra kwargs
        log_data.update(kwargs)
        
        self.logger.log(level, json.dumps(log_data, default=str))
    
    def debug(self, event: str, message: str = "", **kwargs):
        """Log debug level."""
        self._log(logging.DEBUG, event, message, **kwargs)
    
    def info(self, event: str, message: str = "", **kwargs):
        """Log info level."""
        self._log(logging.INFO, event, message, **kwargs)
    
    def warning(self, event: str, message: str = "", **kwargs):
        """Log warning level."""
        self._log(logging.WARNING, event, message, **kwargs)
    
    def error(self, event: str, message: str = "", **kwargs):
        """Log error level."""
        self._log(logging.ERROR, event, message, **kwargs)
    
    def critical(self, event: str, message: str = "", **kwargs):
        """Log critical level."""
        self._log(logging.CRITICAL, event, message, **kwargs)
    
    def log_flow(self, flow_id: str, node_id: str, event: str, **kwargs):
        """Log flow-related event with trace context."""
        trace_id = get_trace_id()
        self.info(event, trace_id=trace_id, flow_id=flow_id, node_id=node_id, **kwargs)
    
    def log_node(self, flow_id: str, node_id: str, node_name: str, 
                 event: str, **kwargs):
        """Log node-related event."""
        trace_id = get_trace_id()
        self.info(event, trace_id=trace_id, flow_id=flow_id, 
                  node_id=node_id, node_name=node_name, **kwargs)
    
    def log_llm(self, flow_id: str, event: str, **kwargs):
        """Log LLM-related event."""
        trace_id = get_trace_id()
        self.info(event, trace_id=trace_id, flow_id=flow_id, **kwargs)
    
    def log_tool(self, flow_id: str, tool_name: str, event: str, **kwargs):
        """Log tool-related event."""
        trace_id = get_trace_id()
        self.info(event, trace_id=trace_id, flow_id=flow_id, 
                  tool_name=tool_name, **kwargs)
    
    def log_memory(self, flow_id: str, operation: str, **kwargs):
        """Log memory operation."""
        trace_id = get_trace_id()
        self.info(f"memory_{operation}", trace_id=trace_id, flow_id=flow_id, **kwargs)


class JsonFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        # If message is already JSON, return it
        try:
            data = json.loads(record.getMessage())
            return json.dumps(data)
        except (json.JSONDecodeError, TypeError):
            pass
        
        # Otherwise, format as JSON
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        
        # Add extra fields
        if hasattr(record, 'trace_id') and record.trace_id:
            log_data['trace_id'] = record.trace_id
        
        if hasattr(record, 'flow_id') and record.flow_id:
            log_data['flow_id'] = record.flow_id
        
        if hasattr(record, 'node_id') and record.node_id:
            log_data['node_id'] = record.node_id
        
        # Add exception info
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_data, default=str)


# Global structured logger
structured_logger = StructuredLogger("neurocore")


# =============================================================================
# Auto-instrumentation Helpers
# =============================================================================

def instrument_flow_runner():
    """
    Instrument the flow runner for automatic tracing.
    
    This should be called during app initialization.
    """
    from core.flow_runner import FlowRunner
    
    # Wrap the run method
    original_run = FlowRunner.run
    
    @wraps(original_run)
    async def traced_run(self, initial_input: dict, start_node_id: str = None, timeout: float = None):
        # Create or propagate trace context
        trace_id = initial_input.get("_trace_id") if isinstance(initial_input, dict) else None
        with create_trace_context(trace_id) as ctx:
            ctx.attributes["flow_id"] = self.flow_id
            ctx.attributes["start_node"] = start_node_id
            
            with ctx.add_span("flow_execution", attributes={
                "flow_id": self.flow_id,
                "timeout": timeout,
            }):
                structured_logger.log_flow(self.flow_id, "system", "flow_start", 
                                           start_node=start_node_id, timeout=timeout)
                
                start_time = time.time()
                try:
                    result = await original_run(self, initial_input, start_node_id, timeout)
                    elapsed_ms = (time.time() - start_time) * 1000
                    
                    metrics.timing("flow.latency", elapsed_ms, {"flow_id": self.flow_id})
                    metrics.increment("flow.completed", tags={"flow_id": self.flow_id})
                    
                    structured_logger.log_flow(self.flow_id, "system", "flow_complete",
                                               duration_ms=elapsed_ms)
                    return result
                except Exception as e:
                    elapsed_ms = (time.time() - start_time) * 1000
                    metrics.timing("flow.latency", elapsed_ms, {"flow_id": self.flow_id})
                    metrics.increment("flow.failed", tags={"flow_id": self.flow_id})
                    structured_logger.log_flow(self.flow_id, "system", "flow_error",
                                               error=str(e), duration_ms=elapsed_ms)
                    raise
    
    FlowRunner.run = traced_run


def instrument_llm_calls():
    """
    Instrument LLM calls for automatic tracing and metrics.
    
    This should be called during app initialization.
    """
    from core.llm import LLMBridge, get_shared_client
    
    # Patch the chat_completion method on LLMBridge class
    if hasattr(LLMBridge, 'chat_completion'):
        original_chat_completion = LLMBridge.chat_completion
        
        @wraps(original_chat_completion)
        async def traced_chat_completion(self, *args, **kwargs):
            ctx = get_or_create_trace_context()
            span = ctx.add_span("llm_call", SpanKind.CLIENT, attributes={
                "model": kwargs.get("model", args[1] if len(args) > 1 else None),
            })
            
            start_time = time.time()
            try:
                result = await original_chat_completion(self, *args, **kwargs)
                elapsed_ms = (time.time() - start_time) * 1000
                
                # Extract token usage if available
                if isinstance(result, dict):
                    tokens = result.get("usage", {})
                    if tokens:
                        metrics.increment("llm.tokens_used", tokens.get("total_tokens", 0))
                        metrics.increment("llm.prompt_tokens", tokens.get("prompt_tokens", 0))
                        metrics.increment("llm.completion_tokens", tokens.get("completion_tokens", 0))
                
                metrics.timing("llm.latency", elapsed_ms)
                metrics.increment("llm.calls_success")
                
                span.finish()
                return result
            except Exception as e:
                elapsed_ms = (time.time() - start_time) * 1000
                metrics.timing("llm.latency", elapsed_ms)
                metrics.increment("llm.calls_failed")
                span.finish(status="error", error=str(e))
                raise
        
        LLMBridge.chat_completion = traced_chat_completion
    
    # Also patch get_shared_client to return instrumented client
    original_get_shared_client = get_shared_client
    
    async def instrumented_get_shared_client(*args, **kwargs):
        client = await original_get_shared_client(*args, **kwargs)
        # The client is an httpx.AsyncClient, we can't easily instrument it
        # Instead, we rely on patching LLMBridge methods which is done above
        return client
    
    # Note: We don't replace get_shared_client as it returns the raw httpx client
    # The actual LLM calls go through LLMBridge.chat_completion which is patched above


def instrument_tools():
    """
    Instrument tool execution for automatic tracing and metrics.
    """
    from modules.tools.node import ToolDispatcherExecutor
    
    if hasattr(ToolDispatcherExecutor, 'receive'):
        original_receive = ToolDispatcherExecutor.receive
        
        @wraps(original_receive)
        async def traced_receive(self, input_data, config=None):
            ctx = get_or_create_trace_context()
            tool_calls = input_data.get("tool_calls", []) if isinstance(input_data, dict) else []
            
            with ctx.add_span("tool_execution", attributes={
                "tool_count": len(tool_calls),
            }):
                start_time = time.time()
                try:
                    result = await original_receive(self, input_data, config)
                    elapsed_ms = (time.time() - start_time) * 1000
                    
                    metrics.timing("tool.latency", elapsed_ms)
                    metrics.increment("tool.calls_success", len(tool_calls))
                    
                    return result
                except Exception as e:
                    elapsed_ms = (time.time() - start_time) * 1000
                    metrics.timing("tool.latency", elapsed_ms)
                    metrics.increment("tool.calls_failed", len(tool_calls))
                    raise
        
        ToolDispatcherExecutor.receive = traced_receive


# =============================================================================
# Dashboard Data Endpoint
# =============================================================================

def get_dashboard_data() -> Dict[str, Any]:
    """
    Get data for observability dashboard.
    
    Returns:
        Dict with metrics, traces, and system info
    """
    return {
        "metrics": metrics.get_all_metrics(),
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }

