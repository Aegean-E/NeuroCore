# Concurrency Model Documentation

## Overview

NeuroCore uses a hybrid concurrency model with both synchronous and asynchronous code. This document outlines the lock ordering rules to prevent deadlocks.

## Lock Types Used

### 1. threading.RLock
Used in synchronous code paths:
- `core/flow_manager.py` - FlowManager.lock
- `core/module_manager.py` - ModuleManager.lock
- `core/observability.py` - Metrics._lock

### 2. asyncio.Lock
Used in asynchronous code paths:
- `core/llm.py` - _client_lock
- `core/flow_runner.py` - _cache_lock
- `modules/chat/sessions.py` - _async_lock
- `modules/reasoning_book/service.py` - _lock

### 3. No Lock
- `core/debug.py` - DebugLogger (explicitly not thread-safe for compound operations)

## Lock Ordering Rules

To prevent deadlocks, follow these rules:

### Rule 1: Never mix threading and asyncio locks in the same code path
- If you hold a threading.RLock, never await an asyncio operation that might try to acquire an asyncio.Lock
- If you hold an asyncio.Lock, never call synchronous code that might try to acquire a threading.RLock

### Rule 2: Acquire locks in consistent order
When multiple locks must be acquired:
1. Always acquire threading.RLock before asyncio.Lock (never the reverse)
2. If multiple threading.RLocks needed, acquire in alphabetical order by variable name
3. If multiple asyncio.Locks needed, acquire in alphabetical order by variable name

### Rule 3: Use context managers
Always use `with` for threading locks and `async with` for asyncio locks:
```python
# Good
with self.lock:
    # do work

# Good  
async with self._async_lock:
    # do async work

# Bad - will cause deadlock
with self.lock:
    await something()  # Don't await while holding threading lock
```

### Rule 4: Keep critical sections small
Minimize the time locks are held to reduce deadlock window.

## Flow Manager and Module Manager

Both use threading.RLock because:
1. They are accessed from synchronous FastAPI route handlers
2. Their operations involve file I/O which is synchronous
3. The asyncio-based FlowRunner calls them during initialization (synchronous)

## Best Practices

1. **Document lock hierarchies** in each module's docstring
2. **Use RLock** instead of Lock when the same thread might acquire the lock multiple times
3. **Add timeout** to lock acquisitions when possible (use `Lock(timeout=...)` if available)
4. **Log warnings** when operations take longer than 1 second (potential lock contention)

## Testing for Deadlocks

Run the concurrency stress tests:
```bash
pytest tests/test_core_concurrency.py -v
```

These tests verify:
- Multiple threads can access FlowManager concurrently
- Multiple coroutines can access async locks
- No deadlocks occur under load

