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
- **FlowRunner**: Executes flows using Kahn's topological sort and bridge group logic.
- **ModuleManager**: Handles hot-loading and unloading of extension modules.
- **FlowManager**: Manages persistence and CRUD for AI flows.
- **SettingsManager**: Centralized configuration and metrics tracking.

### 2.3 Module Layer
Independent, self-contained directories under `modules/` that extend system functionality.
- Each module implements the `NodeExecutor` interface (`receive` and `send`).
- Includes LLM integration, memory systems, tool sandboxes, and more.

### 2.4 Data Layer
- **SQLite**: Stores relational metadata, structured memory, and conversation history.
- **FAISS**: A vector database for high-performance similarity search (RAG).
- **JSON**: Used for local configuration and flow definitions (`ai_flows.json`).

---

## 3. Flow Engine (DAG Execution)

NeuroCore executes workflows as **Directed Acyclic Graphs (DAGs)**.

### 3.1 Execution Workflow
1. **Topological Sort**: Kahn's algorithm determines the execution sequence.
2. **Bridge Groups**: Parallel components are grouped using BFS to enable implicit data sharing.
3. **Node Execution**: Each node processes input via its `receive` method and produces output via `send`.
4. **Conditional Routing**: Dynamic branching is driven by `_route_targets`.
5. **Loop Guard**: A safety counter (`max_node_loops`) prevents infinite loops.

### 3.2 Bridge System
Bridges create implicit bidirectional connections between nodes in the same "bridge group." This allows nodes like **Memory Recall**, **System Prompt**, and **LLM Core** to share a unified execution context without explicit wires.

---

## 4. Concurrency & Thread Safety

NeuroCore uses a hybrid concurrency model:
- **`threading.RLock`**: Used for synchronous shared state (Settings, Metrics).
- **`asyncio.Lock`**: Used for asynchronous resources (LLM clients, ChatSessions).

For detailed locking rules, see [./docs/CONCURRENCY.md](./docs/CONCURRENCY.md).

---

## 5. Technology Stack

| Component | Technology |
|-----------|------------|
| Backend | Python 3.12+ (FastAPI) |
| Frontend | HTMX + Tailwind CSS |
| Database | SQLite + FAISS |
| LLM | OpenAI API Compatible (via httpx) |
| Validation | Pydantic 2.10+ |
| Testing | pytest |
