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
│                                    │                                    │
│                                    ▼                                    │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐   │
│  │ LLMBridge   │  │ DebugLogger │  │ Routers     │  │ Dependencies│   │
│  │ (API Client)│  │ (Tracing)   │  │ (HTTP API)  │  │ (DI)        │   │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘   │
└────────────────────────────────────┬──────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           MODULE LAYER (Extensible)                         │
│                                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐      │
│  │ llm_module  │  │system_prompt│  │   memory    │  │   tools     │      │
│  │   (Core)    │  │  (Context)  │  │  (Vector)   │  │ (Sandbox)   │      │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐      │
│  │    chat     │  │   logic     │  │   planner   │  │ reflection  │      │
│  │   (I/O)     │  │ (Control)   │  │ (Planning)  │  │ (Quality)   │      │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘      │
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
- **UI**: Built with FastAPI, HTMX, and Tailwind CSS to provide a responsive, real-time interface.
- **Flow Editor**: A visual canvas for designing DAG-based AI workflows.
- **Chat Interface**: Multi-platform support (Web UI, Telegram).

### 2.2 Core Layer
The "brain" of the framework, handling orchestration, configuration, and foundational services.
- **FlowRunner**: Executes flows using Kahn's topological sort and bridge group logic. Supports `timeout`, `raise_errors`, and `episode_id` parameters for episode persistence.
- **ModuleManager**: Handles hot-loading and unloading of extension modules. Uses `_loaded_once` set to distinguish initial loads from hot-reloads (prevents premature `sys.modules` flush).
- **FlowManager**: Manages persistence and CRUD for AI flows.
- **SettingsManager**: Centralized, thread-safe configuration. Writes via atomic tempfile+rename. Validates `module_allowlist` on save.
- **SessionManager / EpisodeState**: Persists chat sessions and long-running episode state (plan, current step, completed steps, phase) to `data/episodes/`. Phases: `PHASE_PLANNING`, `PHASE_EXECUTING`, `PHASE_REPLANNING`, `PHASE_COMPLETED`, `PHASE_FAILED`, `PHASE_PAUSED`.
- **PlanHelper** (`core/planner_helpers.py`): Shared utility consolidating plan dependency graph logic previously scattered across `PlannerExecutor` and `PlanStepTracker`. Provides `build_dependency_graph`, `detect_circular_dependencies`, `get_executable_steps`, `generate_plan_context`, and `validate_dependencies`.
- **FlowContext** (`core/flow_context.py`): Pydantic-based type-safe flow payload model. Provides runtime validation and IDE-friendly type hints.
- **FlowData** (`core/flow_data.py`): `TypedDict`-based schema for flow payloads. Includes helper functions (`get_messages`, `set_plan`, etc.) and backward-compatible migration utilities.
- **Error Hierarchy** (`core/errors.py`): Typed exception classes — `NeuroCoreError`, `LLMError`, `LLMTimeoutError`, `LLMHTTPError`, `LLMResponseError`, `ToolError`, `ToolExecutionError`, `ToolTimeoutError`, `SandboxSecurityError`, `FlowError`, `FlowNotFoundError`, `FlowValidationError`, `NodeExecutionError`, `ModuleError`, `ModuleNotFoundError`, `ModuleLoadError`, `MemoryError`, `MemoryConsolidationError`.

### 2.3 Module Layer
Independent, self-contained directories under `modules/` that extend system functionality.
- Each module implements the `NodeExecutor` interface (`receive` and `send`).
- Includes LLM integration, memory systems, tool sandboxes, and more.

### 2.4 Data Layer
- **SQLite**: Stores relational metadata, structured memory, and conversation history.
- **FAISS**: A vector database for high-performance similarity search (RAG).
- **JSON**: Used for local configuration and flow definitions (`ai_flows.json`, `chat_sessions.json`, `data/reasoning_book.json`).
- **JSONL**: `data/execution_trace.jsonl` stores per-node execution traces for debugging.
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
1. **Topological Sort**: Kahn's algorithm determines the execution sequence.
2. **Bridge Groups**: Parallel components are grouped using BFS to enable implicit data sharing.
3. **Node Execution**: Each node processes input via its `receive` method and produces output via `send`.
4. **Conditional Routing**: Dynamic branching is driven by `_route_targets`.
5. **Loop Guard**: A safety counter (`max_node_loops`) prevents infinite loops.
6. **Timeout**: Optional per-flow `timeout` parameter wraps execution in `asyncio.wait_for`.
7. **Error Mode**: `raise_errors=True` propagates node exceptions instead of returning error dicts.
8. **Episode Persistence**: `episode_id` restores `EpisodeState` (plan, current step, completed steps) from `data/episodes/` before execution begins.

### 3.2 Bridge System
Bridges create implicit bidirectional connections between nodes in the same "bridge group." This allows nodes like **Memory Recall**, **System Prompt**, and **LLM Core** to share a unified execution context without explicit wires.

---

## 4. Concurrency & Thread Safety

NeuroCore uses a hybrid concurrency model:
- **`threading.RLock`**: Used for synchronous shared state (FlowManager, ModuleManager, SettingsManager, SessionPersistenceManager).
- **`threading.Lock`**: Used for single-level synchronous guards (Metrics, SessionManager, singleton init).
- **`asyncio.Lock`**: Used for asynchronous resources (LLM clients, FlowRunner cache per event loop, ChatSessions, ReasoningBook).

For detailed locking rules, see [./CONCURRENCY.md](./CONCURRENCY.md).

---

## 5. Technology Stack

| Component | Technology |
|-----------|------------|
| Backend | Python 3.12+, FastAPI 0.115+, Uvicorn 0.32+ |
| Frontend | HTMX, Vanilla JS, TailwindCSS (CDN), Jinja2 3.1+ |
| Vector DB | FAISS `IndexFlatIP` + L2 normalization |
| Relational DB | SQLite (WAL mode, FTS5 full-text search) |
| HTTP Client | HTTPX 0.28+ (async, connection pooling) |
| LLM Integration | OpenAI-compatible API |
| Data Validation | Pydantic 2.10+ |
| Testing | pytest, pytest-asyncio (`asyncio_mode = "auto"`), pytest-httpx, pytest-cov |
| Deployment | Docker + docker-compose |
| Linting | Ruff (configured in `pyproject.toml`) |

## 6. Key Configuration Settings

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

`module_allowlist` is a security control: when non-empty, only listed module IDs can be hot-loaded by `ModuleManager`. `ui_wide_mode` and `ui_show_footer` control the web UI layout. `debug_mode` enables per-node execution tracing via `DebugLogger` and triggers `importlib.reload()` on every executor load.
