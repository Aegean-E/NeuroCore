# NeuroCore System Architecture

This document provides a high-level overview of the NeuroCore system architecture, its core layers, and the execution engine.

---

## 1. Architectural Overview

NeuroCore follows a layered architecture, separating the presentation, core logic, extensible modules, and data persistence.

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                              PRESENTATION LAYER                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │  Chat UI    │  │ Flow Editor │  │ Memory      │  │  Module Dashboard   │ │
│  │  (HTMX)     │  │ (Canvas)    │  │ Browser     │  │  (Settings)         │ │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘ │
└─────────┼────────────────┼────────────────┼────────────────────┼────────────┘
          │                │                │                    │
          └────────────────┴────────────────┴────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CORE LAYER                                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐  │
│  │ FlowRunner  │  │FlowManager  │  │ModuleManager│  │  SettingsManager│  │
│  │ (Execution) │  │ (Storage)   │  │ (Lifecycle) │  │  (Config)       │  │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └────────┬────────┘  │
│         │                │                │                  │           │
│         └────────────────┴────────────────┴──────────────────┘           │
│                                    │                                      │
│                                    ▼                                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐   │
│  │ LLMBridge   │  │Observability│  │ Routers     │  │ Dependencies│   │
│  │ (API Client)│  │(Trace/Metr.)│  │ (HTTP API)  │  │ (DI)        │   │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘   │
└────────────────────────────────────┬──────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        MODULE LAYER (17 Modules)                            │
│                                                                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │llm_module│  │  system  │  │  memory  │  │  tools   │  │  chat    │   │
│  │  (Core)  │  │  _prompt │  │ (Vector) │  │(Sandbox) │  │  (I/O)   │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │  logic   │  │ planner  │  │reflection│  │knowledge │  │ agent_   │   │
│  │(Control) │  │(Planning)│  │(Quality) │  │  _base   │  │  loop    │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │messaging │  │ calendar │  │  skills  │  │reasoning │  │ browser  │   │
│  │  _bridge │  │ (Events) │  │(Instruct)│  │  _book   │  │  _auto   │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
│  ┌──────────┐  ┌──────────┐                                               │
│  │ memory_  │  │annotatio │                                               │
│  │ browser  │  │   ns     │                                               │
│  └──────────┘  └──────────┘                                               │
└────────────────────────────────────┬──────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DATA LAYER                                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐      │
│  │  SQLite     │  │   FAISS     │  │   JSON      │  │   File      │      │
│  │ (Metadata)  │  │ (Vectors)   │  │ (Config)    │  │  Storage    │      │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘      │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Core Layers

### 2.1 Presentation Layer
- **UI**: Built with FastAPI, HTMX, and Tailwind CSS to provide a responsive, real-time interface (36 Jinja2 templates).
- **Flow Editor**: A visual canvas for designing DAG-based AI workflows with version history and rollback.
- **Chat Interface**: Web UI with real-time streaming and thinking trace display.

### 2.2 Core Layer
The "brain" of the framework, handling orchestration, configuration, and foundational services.
- **FlowRunner** (`core/flow_runner.py`): Executes flows using Kahn's topological sort and bridge group logic. Supports `timeout`, `raise_errors`, and `episode_id` parameters for episode persistence. Per-event-loop executor cache (max 100 entries) avoids re-instantiation.
- **ModuleManager** (`core/module_manager.py`): Handles hot-loading and unloading of extension modules. Uses `_loaded_once` set to distinguish initial loads from hot-reloads (prevents premature `sys.modules` flush). Enforces `module_allowlist` from settings.
- **FlowManager** (`core/flow_manager.py`): Manages persistence and CRUD for AI flows, including version history (up to 20 versions per flow stored in `ai_flows_versions.json`).
- **SettingsManager** (`core/settings.py`): Centralized, thread-safe configuration. Writes via atomic tempfile+rename. Validates all settings on save.
- **SessionManager / EpisodeState** (`core/session_manager.py`): Persists chat sessions and long-running episode state (plan, current step, completed steps, phase) to `data/episodes/`. Phases: `PHASE_PLANNING`, `PHASE_EXECUTING`, `PHASE_REPLANNING`, `PHASE_COMPLETED`, `PHASE_FAILED`, `PHASE_PAUSED`.
- **PlanHelper** (`core/planner_helpers.py`): Shared utility consolidating plan dependency graph logic. Provides `build_dependency_graph`, `detect_circular_dependencies`, `get_executable_steps`, `generate_plan_context`, and `validate_dependencies`.
- **FlowContext** (`core/flow_context.py`): Pydantic-based type-safe flow payload model. Provides runtime validation, `messaging_platform`/`messaging_reply_to` fields, and IDE-friendly type hints.
- **FlowData** (`core/flow_data.py`): `TypedDict`-based schema for flow payloads. Includes helper functions (`get_messages`, `set_plan`, etc.) and backward-compatible migration utilities. Declares all reserved keys including `_messaging_platform` and `_messaging_reply_to`.
- **Error Hierarchy** (`core/errors.py`): 14 typed exception classes — `NeuroCoreError`, `LLMError`, `LLMTimeoutError`, `LLMHTTPError`, `LLMResponseError`, `ToolError`, `ToolExecutionError`, `ToolTimeoutError`, `SandboxSecurityError`, `FlowError`, `FlowNotFoundError`, `FlowValidationError`, `NodeExecutionError`, `ModuleError`, `ModuleNotFoundError`, `ModuleLoadError`, `MemoryError`, `MemoryConsolidationError`.
- **Observability** (`core/observability.py`): Distributed tracing (span-based with parent-child relationships), metrics collection (counters, gauges, histograms with p50/p95/p99), and structured JSON logging. Metrics counters are persisted across restarts.

### 2.3 Module Layer
17 independent, self-contained directories under `modules/` extending system functionality.
- Each module implements the `NodeExecutor` interface (`receive` and `send`).
- Modules are hot-loadable — enable/disable without restarting the server.
- See [MODULE_GUIDE.md](./MODULE_GUIDE.md) for the full development guide.

### 2.4 Data Layer
- **SQLite**: Stores relational metadata, structured memory, and conversation history (WAL mode, FTS5 full-text search).
- **FAISS**: `IndexFlatIP` with L2 normalization for high-performance similarity search (memory + knowledge base).
- **JSON**: Local configuration and flow definitions (`ai_flows.json`, `ai_flows_versions.json`, `chat_sessions.json`, `data/reasoning_book.json`).
- **JSONL**: `data/execution_trace.jsonl` stores per-node execution traces for debugging (append-only, written only when `debug_mode=true`).
- **Episodes**: `data/episodes/` stores serialized `EpisodeState` objects for long-running agent tasks.

### 2.5 Scientific Schemas (`core/schemas/`)

Domain models for academic research management, usable as structured output targets:

| Schema | File | Purpose |
|--------|------|---------|
| **Hypothesis** | `hypothesis.py` | Scientific hypotheses with variables and confidence levels |
| **Article** | `article.py` | Academic articles with bibliographic info and citations |
| **Finding** | `finding.py` | Research findings with evidence linking |
| **StudyDesign** | `study_design.py` | Scientific study designs with methodology |

---

## 3. Flow Engine (DAG Execution)

NeuroCore executes workflows as **Directed Acyclic Graphs (DAGs)**.

### 3.1 Execution Workflow
1. **[Optional] Episode Restore**: If `episode_id` is provided, `EpisodeState` is loaded from `data/episodes/` and injected into `initial_input`.
2. **Topological Sort**: Kahn's algorithm determines the execution sequence.
3. **Bridge Groups**: Parallel components are grouped using BFS to enable implicit data sharing.
4. **Node Execution**: Each node processes input via its `receive` method and produces output via `send`. Messages list is deep-copied before each node to prevent cross-node mutation.
5. **Conditional Routing**: Dynamic branching is driven by `_route_targets`.
6. **Loop Guard**: A safety counter (`max_node_loops`, default 100, max 1000) prevents infinite loops.
7. **Timeout**: Optional per-flow `timeout` parameter wraps execution in `asyncio.wait_for`.
8. **Error Mode**: `raise_errors=True` propagates node exceptions instead of returning error dicts.

### 3.2 Bridge System
Bridges create implicit bidirectional connections between nodes in the same "bridge group," enabling Memory Recall, System Prompt, and LLM Core to share a unified execution context without explicit wires.

### 3.3 Input Node Routing
The flow runner uses an `input_node_map` to route incoming data to the correct source node:

| Source | Target Node |
|--------|------------|
| `chat` | `chat_input` |
| `discord` | `discord_input` (legacy) |
| `messaging` | `messaging_input` |

---

## 4. Messaging Bridge

The `messaging_bridge` module provides a unified interface for all messaging platforms through a single pair of nodes:

- **`messaging_input`**: Receives messages from any platform; filters by platform list in config.
- **`messaging_output`**: Routes replies back to the originating platform or to configured proactive recipients.

### 4.1 Supported Platforms

| Platform | Mechanism | Notes |
|----------|-----------|-------|
| Telegram | HTTP long-polling | 3072-char chunking |
| Discord | WebSocket Gateway v10 | 1900-char chunking, heartbeat loop |
| Signal | HTTP polling (signal-cli REST) | 1800-char chunking |
| WhatsApp | Webhook (Evolution API) | 4000-char chunking, no polling thread |

### 4.2 Reserved Flow Keys

| Key | Source | Purpose |
|-----|--------|---------|
| `_messaging_platform` | MessagingInputExecutor | Platform that originated the message |
| `_messaging_reply_to` | MessagingInputExecutor | Sender address (chat_id/channel/phone/JID) |

### 4.3 MESSAGING_PLATFORMS Registry
`modules/messaging_bridge/node.py` defines `MESSAGING_PLATFORMS` as the single source of truth for all supported platforms. Adding a new platform requires only appending one entry plus implementing a bridge class.

---

## 5. Concurrency & Thread Safety

NeuroCore uses a hybrid concurrency model:
- **`threading.RLock`**: Used for synchronous shared state (FlowManager, ModuleManager, SettingsManager, SessionPersistenceManager).
- **`threading.Lock`**: Used for single-level synchronous guards (Metrics, SessionManager instance, `_init_lock` singleton guard).
- **`asyncio.Lock`**: Used for asynchronous resources (LLM clients, FlowRunner cache per event loop, ChatSessions via `asyncio.to_thread`, ReasoningBook).

For detailed locking rules, see [./CONCURRENCY.md](./CONCURRENCY.md).

---

## 6. Technology Stack

| Component | Technology |
|-----------|------------|
| Backend | Python 3.12+, FastAPI 0.115+, Uvicorn 0.32+ |
| Frontend | HTMX, Vanilla JS, TailwindCSS (CDN), Jinja2 3.1+ |
| Vector DB | FAISS `IndexFlatIP` + L2 normalization |
| Relational DB | SQLite (WAL mode, FTS5 full-text search) |
| HTTP Client | HTTPX 0.28+ (async, connection pooling) |
| WebSocket | websockets 12.0+ (Discord Gateway, custom protocols) |
| LLM Integration | OpenAI-compatible API |
| Data Validation | Pydantic 2.10+ |
| Testing | pytest, pytest-asyncio (`asyncio_mode = "auto"`), pytest-httpx, pytest-cov |
| Deployment | Docker + docker-compose |
| Linting | Ruff (configured in `pyproject.toml`) |

---

## 7. Key Configuration Settings

All runtime configuration lives in `settings.json`. The `SettingsManager` provides atomic reads and writes protected by `threading.RLock`.

```python
DEFAULT_SETTINGS = {
    "llm_api_url":        "http://localhost:1234/v1",
    "llm_api_key":        "",
    "embedding_api_url":  "",
    "default_model":      "local-model",
    "embedding_model":    "",
    "active_ai_flows":    [],
    "temperature":        0.7,
    "max_tokens":         2048,
    "debug_mode":         False,
    "ui_wide_mode":       False,
    "ui_show_footer":     True,
    "request_timeout":    60.0,
    "max_node_loops":     100,
    "module_allowlist":   [],   # empty = allow all modules
}
```

`module_allowlist` is a security control: when non-empty, only listed module IDs can be hot-loaded by `ModuleManager`. `debug_mode` enables per-node execution tracing via `observability` and triggers `importlib.reload()` on every executor load.

---

## 8. Current Scale (as of March 2026)

| Metric | Value |
|--------|-------|
| Active modules | 17 |
| Node executors | 27 |
| HTTP routes (core) | 45 |
| Test files | 71 |
| Test cases | 1,051+ |
| Web templates | 36 |
| Built-in tools | 23 (16 standard + 7 RLM) |
| Python files | 165+ |
