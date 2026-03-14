# NeuroCore Project Exhaustive Analysis

## Executive Summary

NeuroCore is a **modular, extensible AI orchestration platform** built with Python/FastAPI. It provides a visual flow-based system for building autonomous AI agents with memory, tools, knowledge bases, and multi-platform integrations. The platform features a sophisticated hybrid concurrency model, comprehensive observability infrastructure, and scientific research schemas for academic article management.

---

## 1. Architecture Overview

### 1.1 System Architecture

```
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
│                                    │                                    │
│                                    ▼                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │                    OBSERVABILITY LAYER                              │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐   │  │
│  │  │ Distributed │  │   Metrics   │  │  Structured Logging     │   │  │
│  │  │   Tracing   │  │  Collection │  │  (JSON Format)          │   │  │
│  │  └─────────────┘  └─────────────┘  └─────────────────────────┘   │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                                    │                                    │
│                                    ▼                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │                   STRUCTURED OUTPUT LAYER                           │  │
│  │  ┌─────────────────────────────────────────────────────────────────┐ │  │
│  │  │ Pydantic Schema Enforcement │ Retry Logic │ Provider Support │ │  │
│  │  └─────────────────────────────────────────────────────────────────┘ │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           MODULE LAYER (16 Modules)                         │
│                                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐      │
│  │ llm_module  │  │system_prompt│  │   memory    │  │   tools     │      │
│  │   (Core)    │  │  (Context)  │  │  (Vector)   │  │ (Sandbox)   │      │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐      │
│  │    chat     │  │   logic     │  │   planner   │  │ reflection  │      │
│  │   (I/O)     │  │ (Control)   │  │ (Planning)  │  │ (Quality)   │      │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐      │
│  │knowledge_base│  │reasoning_book│ │   skills    │  │   calendar  │      │
│  │   (RAG)     │  │ (Thoughts)  │  │ (Templates) │  │ (Events)    │      │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐      │
│  │  telegram   │  │memory_browser│ │ annotations │  │ agent_loop  │      │
│  │  (Bridge)   │  │   (UI)      │  │  (Docs)     │  │ (Autonomous)│      │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘      │
└─────────────────────────────────────────────────────────────────────────────┘
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

### 1.2 Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Backend** | Python 3.12+ | Core runtime |
| **Web Framework** | FastAPI | HTTP API & routing |
| **Frontend** | HTMX + Tailwind CSS | Dynamic UI |
| **Templating** | Jinja2 | HTML generation |
| **Database** | SQLite + FAISS | Data & vector storage |
| **LLM Client** | httpx | Async API calls |
| **Concurrency** | asyncio | Async execution |
| **Data Validation** | Pydantic 2.10+ | Schema enforcement |
| **Testing** | pytest | Test framework |

### 1.3 Project Structure

```
NeuroCore/
├── main.py                    # FastAPI app entry point, lifespan management
├── core/                      # Core framework (15 files, ~36K LOC total project)
│   ├── flow_runner.py        # DAG execution engine (Kahn's sort, bridge groups, episode support)
│   ├── flow_manager.py       # Flow CRUD operations (JSON persistence)
│   ├── module_manager.py     # Dynamic module loading (_loaded_once, allowlist)
│   ├── llm.py                # LLM API client (OpenAI-compatible, async)
│   ├── settings.py           # Thread-safe config (RLock, atomic write)
│   ├── observability.py      # Tracing, metrics, structured JSON logging
│   ├── structured_output.py  # Pydantic schema enforcement (retry logic)
│   ├── session_manager.py    # Chat session + EpisodeState persistence
│   ├── debug.py              # Structured debug logging
│   ├── routers.py            # 40+ HTTP endpoints
│   ├── dependencies.py       # FastAPI dependency injection (LLMBridge)
│   ├── errors.py             # Typed exception hierarchy
│   ├── flow_data.py          # FlowData TypedDict + helper functions
│   ├── flow_context.py       # FlowContext Pydantic model
│   ├── planner_helpers.py    # PlanHelper: dependency graphs, cycle detection
│   └── schemas/              # Domain models for scientific research
│       ├── hypothesis.py
│       ├── article.py
│       ├── finding.py
│       └── study_design.py
├── modules/                  # Extensible modules (16 total)
│   ├── llm_module/          # Core LLM integration
│   ├── system_prompt/       # Context injection (memory, KB, reasoning, plan, skills)
│   ├── memory/              # Vector + SQLite memory (FAISS + arbiter + consolidation)
│   ├── tools/               # Sandbox execution
│   │   ├── library/         # 16 standard built-in tool files
│   │   └── rlm_library/     # 7 RLM tool files
│   ├── chat/                # Chat I/O + session management + compaction
│   ├── logic/               # Control flow nodes (Delay, Repeater, Conditional, etc.)
│   ├── planner/             # Planning nodes
│   ├── reflection/          # Quality assurance node
│   ├── knowledge_base/      # RAG system (hybrid search, FTS5 + FAISS)
│   ├── reasoning_book/      # Thought logging
│   ├── skills/              # Instruction file management
│   ├── calendar/            # Event management
│   ├── telegram/            # Telegram bot bridge
│   ├── memory_browser/      # Memory management UI
│   ├── annotations/         # Flow documentation nodes
│   └── agent_loop/          # Autonomous execution loop
├── web/
│   └── templates/           # 33 Jinja2 HTML templates
├── data/                    # Runtime data (mutable, excluded from reloader)
│   ├── memory.faiss         # Long-term memory vector index
│   ├── memory.sqlite3       # Long-term memory relational store
│   ├── knowledge_base.faiss # RAG vector index
│   ├── knowledge_base.sqlite3 # RAG relational store
│   ├── reasoning_book.json  # Thought log
│   ├── execution_trace.jsonl # Per-node execution trace (debug)
│   ├── session.json         # Session persistence
│   └── episodes/            # EpisodeState JSON files (long-running tasks)
├── tests/                   # 67 test files, 926 tests
├── docs/                    # Documentation
├── chat_sessions.json       # Chat session history
├── ai_flows.json            # Saved flow definitions
├── settings.json            # Runtime configuration
├── pyproject.toml           # Project metadata, pytest config, ruff rules
├── requirements.txt         # Runtime dependencies
├── Dockerfile
└── docker-compose.yml
```

---

## 2. Core Components Deep Dive

### 2.1 FlowRunner (`core/flow_runner.py`)

**Purpose**: Executes AI flows as directed acyclic graphs (DAGs) with bridge support

**Key Features**:
- **Bridge Groups**: Bidirectional node connections for shared data
- **Topological Sort**: Kahn's algorithm determines execution order
- **Executor Cache**: Avoids re-importing node classes (max 100 entries)
- **Cycle Detection**: Handles malformed flows gracefully
- **Debug Logging**: Per-node execution tracing
- **Automatic Context Propagation**: Preserves conversation history

**`run()` Signature**:
```python
async def run(
    self,
    initial_input: dict,
    start_node_id: str = None,
    timeout: float = None,
    raise_errors: bool = False,
    episode_id: str = None,
) -> dict
```
- `timeout`: wraps execution in `asyncio.wait_for`; returns `{"error": "timeout"}` on expiry
- `raise_errors`: if True, re-raises node exceptions instead of returning error dicts
- `episode_id`: restores `EpisodeState` (plan, current_step, completed_steps) from `data/episodes/` before execution

**Execution Flow**:
```python
1. [Optional] Load EpisodeState from data/episodes/ and inject into initial_input
2. Load flow definition (nodes, connections, bridges)
3. Build bridge groups using BFS (connected components)
4. Compute topological execution order (Kahn's algorithm)
5. For each node in order:
   a. Collect input from upstream nodes (or initial_input for source nodes)
   b. Merge bridge group data
   c. Deep-copy messages list to prevent cross-node mutation
   d. Execute node.receive() → node.send()
   e. Auto-propagate `messages` key if output omits it
   f. Store output for downstream nodes
6. Return final output dict
```

**Bridge System**:
- Bridges create implicit dependencies between nodes
- All nodes in a bridge group share the same input from earlier nodes in the chain are merged data
- Outputs into later nodes
- Enables complex data flow patterns without explicit connections
- Example: Memory Recall → System Prompt → LLM Core (bridged together)

### 2.2 ModuleManager (`core/module_manager.py`)

**Purpose**: Dynamic module loading and lifecycle management

**Key Features**:
- **Hot-Swapping**: Enable/disable modules without restart
- **Auto-Discovery**: Scans `modules/` directory
- **Router Injection**: Dynamically adds FastAPI routes via `app.include_router()`
- **Ordering**: Configurable module sort order
- **Dependency Tracking**: Clears FlowRunner cache on module changes
- **`_loaded_once` Set**: Prevents accidental `sys.modules` flush on first import (critical for test isolation)
- **`DISABLED` Marker**: `DISABLED` file in module dir skips discovery entirely
- **`module_allowlist`**: Runtime security control from `settings.json`
- **`load_error` Isolation**: Load errors stored in `_load_errors` dict only — never written to `module.json`

**Module States**:
```
DISABLED (file marker) → DISCOVERED → ENABLED → LOADED → ACTIVE
                              ↑________↓ (hot-swap supported)
```

### 2.3 FlowManager (`core/flow_manager.py`)

**Purpose**: CRUD operations for AI flows

**Key Features**:
- **Default Flow**: Auto-creates on first run
- **Import/Export**: JSON-based flow sharing
- **Active Flows**: Tracks currently running flows
- **Persistence**: JSON file storage (ai_flows.json)

**Default Flow Structure**:
- Chat Input → Memory Recall → System Prompt → LLM Core → Conditional Router
- Tool Dispatcher (conditional branch)
- Chat Output + Telegram Output (parallel)
- Memory Save (side effect)

### 2.4 LLMBridge (`core/llm.py`)

**Purpose**: Unified interface to LLM APIs

**Key Features**:
- **OpenAI Compatible**: Works with local servers (LM Studio, Ollama, etc.)
- **Embeddings**: Separate endpoint support for vector generation
- **Tool Support**: Function calling format
- **Configurable**: Model, temperature, max_tokens
- **Shared Client**: Connection pooling for efficiency

### 2.4b EpisodeState (`core/session_manager.py`)

**Purpose**: Persist long-running agent task state across flow executions

**Key Fields**:
- `episode_id` — unique identifier
- `plan` — list of steps
- `current_step` — current step index
- `completed_steps` — set of completed step indices
- `phase` — lifecycle phase (`PHASE_PLANNING`, `PHASE_EXECUTING`, `PHASE_REPLANNING`, `PHASE_COMPLETED`, `PHASE_FAILED`, `PHASE_PAUSED`)
- `max_context_tokens` — token budget for context window

**Storage**: `data/episodes/{episode_id}.json`

When `FlowRunner.run(episode_id=...)` is called, the episode is loaded and its state injected into `initial_input` before execution begins. After execution, the phase is updated.

### 2.5 Observability (`core/observability.py`)

**Purpose**: Enterprise-grade observability infrastructure

**Components**:

| Component | Description |
|-----------|-------------|
| **Distributed Tracing** | trace_id propagation, span management via contextvars |
| **Metrics Collection** | Counters, timers, gauges with percentile tracking (p50, p95, p99) |
| **Structured Logging** | JSON-formatted logs with trace context |
| **Auto-Instrumentation** | Patches FlowRunner, LLMBridge, ToolDispatcher |

**Key Classes**:
- `TraceContext`: Manages trace spans with parent-child relationships
- `Metrics`: Thread-safe metrics collector using threading.Lock
- `StructuredLogger`: JSON logger with trace_id correlation
- `instrument_flow_runner()`: Patches FlowRunner.run()
- `instrument_llm_calls()`: Patches LLMBridge.chat_completion()
- `instrument_tools()`: Patches ToolDispatcherExecutor.receive()

### 2.6 Structured Output (`core/structured_output.py`)

**Purpose**: Guaranteed schema-valid JSON from LLM calls

**Key Features**:
- **Pydantic Integration**: Uses BaseModel schemas
- **Provider-Specific Support**: OpenAI, Anthropic, Azure JSON schema support
- **Retry Logic**: Up to 3 attempts with validation error injection
- **Conversation Management**: Limits history to 20 messages
- **Timeout Handling**: Configurable request timeout

**Functions**:
```python
async def structured_completion(
    messages: List[dict],
    schema: Type[BaseModel],
    max_retries: int = 3,
    timeout: float = 60.0
) -> BaseModel
```

### 2.6b FlowData and FlowContext

**`core/flow_data.py`** — `FlowData` TypedDict with typed helper functions:
```python
messages = get_messages(data)         # Returns list, never None
plan     = get_plan(data)             # Returns list
data     = set_plan(data, steps)      # Type-safe setter
```

**`core/flow_context.py`** — `FlowContext` Pydantic model for runtime-validated payloads:
```python
ctx = FlowContext(messages=[...], plan_needed=True)
ctx.messages.append({"role": "assistant", "content": "hi"})
```

Both are optional — existing nodes using plain `dict` continue to work.

### 2.6c PlanHelper (`core/planner_helpers.py`)

Consolidates plan dependency logic previously scattered across `PlannerExecutor` and `PlanStepTracker`:
- `build_dependency_graph(plan)` — produces adjacency list + reverse map from step `dependencies` fields
- `detect_circular_dependencies(plan)` — returns `(bool, cycle_path_or_None)`
- `get_executable_steps(plan, completed_steps)` — returns step indices whose dependencies are all satisfied
- `generate_plan_context(plan, current_step, completed_steps)` — formats a human-readable plan string for system prompt injection
- `validate_dependencies(plan)` — validates all dependency indices are in range

### 2.7 Scientific Schemas (`core/schemas/`)

**Purpose**: Domain models for scientific research management

| Schema | File | Purpose |
|--------|------|---------|
| **Hypothesis** | hypothesis.py | Scientific hypotheses with variables, confidence levels |
| **Article** | article.py | Academic articles with bibliographic info, citations |
| **Finding** | finding.py | Research findings with evidence linking |
| **StudyDesign** | study_design.py | Scientific study designs with methodology |

---

## 3. Module Analysis

### 3.1 Core AI Modules

#### 3.1.1 LLM Module (`modules/llm_module/`)

**Node**: `llm_module`

**Configuration Precedence**:
1. Node config (highest)
2. Input data
3. Module defaults (lowest)

**Parameters**:
- `model`: Model identifier
- `temperature`: Randomness (0-2)
- `max_tokens`: Output limit
- `tools`: Available functions
- `tool_choice`: Force tool usage

#### 3.1.2 System Prompt (`modules/system_prompt/`)

**Node**: `system_prompt`

**Context Injection Sources**:
- Memory context (`_memory_context`)
- Knowledge context (`knowledge_context`)
- Reasoning context (`reasoning_context`)
- Plan context (`plan_context`)
- Skills content (`enabled_skills`)

**Tool Integration**:
- Loads available tools from `modules/tools/tools.json`
- Converts to OpenAI function format
- Configurable enabled tools list

#### 3.1.3 Agent Loop (`modules/agent_loop/`)

**Node**: `agent_loop`

**Autonomous Execution**:
- Loops between LLM and tools
- Max iterations with safety limit
- Exponential backoff retry
- Timeout protection
- Full execution trace logging

**Configuration**:
```json
{
    "max_iterations": 10,
    "max_llm_retries": 3,
    "retry_delay": 1.0,
    "tool_error_strategy": "continue",
    "timeout": 120
}
```

### 3.2 Memory System (`modules/memory/`)

#### 3.2.1 Architecture

**Dual Storage**:
- **SQLite**: Metadata, text, relationships
- **FAISS**: Vector embeddings for similarity search

**Memory Types**:
| Type | Description | TTL |
|------|-------------|-----|
| BELIEF | Uncertain information | 30 days |
| FACT | Verified information | None |
| RULE | Behavioral guidelines | None |
| EXPERIENCE | Past interactions | None |
| PREFERENCE | User preferences | None |
| IDENTITY | Self-knowledge | None |

#### 3.2.2 Memory Lifecycle

```
Create → Store (SQLite + FAISS) → Search (FAISS) → Retrieve → Use
   ↓         ↓                      ↓              ↓
   └─────────┴──────────────────────┴──────────────┘
                    ↓
            Consolidation (LLM-based)
                    ↓
            Parent-Child Chaining
```

#### 3.2.3 Consolidation (`modules/memory/consolidation.py`)

**Algorithm**:
1. Fetch candidate memories (active, no parent)
2. Compute similarity matrix (cosine similarity)
3. For pairs with similarity > 0.92:
   - Use LLM to verify semantic equivalence
   - Link older memory as child of newer
   - Remove from FAISS index
4. Run automatically every 24 hours

#### 3.2.4 Memory Arbiter (`modules/memory/arbiter.py`)

**Purpose**: LLM-based memory quality control

**Features**:
- Verifies new memories for consistency
- Detects contradictory memories
- Suggests memory updates

#### 3.2.5 Nodes

| Node | Purpose |
|------|---------|
| `memory_recall` | Searches memory, injects context |
| `memory_save` | Stores interactions with extraction |
| `check_goal` | Retrieves active goals |

### 3.3 Tools System (`modules/tools/`)

#### 3.3.1 Security Architecture

**Sandbox Layers**:
1. **Static Analysis**: Pre-execution code scanning
2. **Import Restrictions**: Block dangerous modules
3. **Restricted Builtins**: Remove dangerous functions
4. **File Access Control**: Directory whitelist
5. **Network Restrictions**: Domain whitelist
6. **Resource Limits**: Timeout, memory, output size

**Blocked Modules**:
```
os, sys, subprocess, socket, multiprocessing, ctypes, 
mmap, pickle, importlib, shutil, pathlib, ...
```

**Allowed Modules**:
```
math, random, datetime, json, re, collections, 
hashlib, uuid, html, urllib.parse, ...
```

#### 3.3.2 Tool Execution Flow

```
LLM generates tool call → ToolDispatcher receives → 
Load tool code → Sandbox.execute() → 
Return result → Add to message history → 
Conditional Router checks for more tools
```

#### 3.3.3 Tool Library (`modules/tools/library/`)

| Tool | Purpose |
|------|---------|
| Calculator | Safe mathematical expression evaluation (AST-based) |
| Weather | Weather information lookup |
| ArXivSearch | Academic paper search |
| CurrencyConverter | Currency conversion |
| WikipediaLookup | Wikipedia search |
| YouTubeTranscript | YouTube subtitle extraction |
| FetchURL | Web content retrieval (with SSRF protection) |
| SendEmail | Email sending (with TLS verification) |
| SetGoal/DeleteGoal/MarkGoalComplete | Goal management |

#### 3.3.4 RLM Tools (`modules/tools/rlm_library/`)

**Purpose**: Memory manipulation tools for reasoning

| Tool | Purpose |
|------|---------|
| GetVariable | Retrieve stored variables |
| SetVariable | Store variables for later use |
| Peek | Examine current state without modification |
| Search | Search memory |
| SetFinal | Mark final response |
| Chunk | Split large outputs |
| SubCall | Call subroutines |

#### 3.3.5 Nodes

| Node | Purpose |
|------|---------|
| `tool_dispatcher` | Executes tool calls from LLM |

### 3.4 Control Flow (`modules/logic/`)

#### 3.4.1 Nodes

| Node | Purpose | Config |
|------|---------|--------|
| `delay_node` | Pause execution | `seconds` |
| `script_node` | Custom Python | `code` |
| `repeater_node` | Background loop | `delay`, `max_repeats` |
| `conditional_router` | Branch logic | `check_field`, `true/false_branches` |
| `trigger_node` | Manual trigger | - |
| `schedule_start_node` | Time-based start | `schedule_time`, `schedule_date` |

#### 3.4.2 Conditional Router Logic

**Check Fields**:
- `tool_calls`: LLM generated tool calls
- `requires_continuation`: More tools pending
- `satisfied`: Reflection evaluation
- `max_tools_per_turn`: Tool limit reached

**Routing**:
- Sets `_route_targets` in output data
- FlowRunner uses targets to determine next nodes
- Supports inverted conditions

### 3.5 Knowledge Base (`modules/knowledge_base/`)

**RAG Pipeline**:
1. Document upload (PDF, TXT, MD)
2. Text extraction and chunking
3. Embedding generation
4. Vector storage (FAISS)
5. Hybrid search (vector + keyword)
6. Context injection

**Node**: `query_knowledge`

### 3.6 Reasoning Book (`modules/reasoning_book/`)

**Purpose**: Log AI's internal thought process

**Storage**: `data/reasoning_book.json`

**Nodes**:
- `reasoning_save`: Save agent response as thought
- `reasoning_load`: Load recent thoughts into context

### 3.7 Chat System (`modules/chat/`)

**Nodes**:
- `chat_input`: Entry point, validates messages
- `chat_output`: Formats final response

**Features**:
- Session management (create, rename, delete)
- Message history with token estimation
- Auto-renaming (LLM-generated title after N turns)
- **Session Compaction**: LLM summarizes older messages when token count exceeds `auto_compact_tokens`; preserves last `compact_keep_last` messages verbatim. Available as `POST /chat/sessions/{id}/compact` and triggered automatically on send.

### 3.8 Telegram Bridge (`modules/telegram/`)

**Nodes**:
- `telegram_input`: Receives messages
- `telegram_output`: Sends responses

**Integration**:
- Webhook-based message receiving
- Bot API for sending
- Supports images and documents

### 3.9 Calendar (`modules/calendar/`)

**Node**: `calendar_watcher`

**Features**:
- Event CRUD
- Reminder notifications
- Recurring events
- ICS import/export

### 3.10 Skills (`modules/skills/`)

**Purpose**: Reusable prompt templates

**Storage**: `modules/skills/data/`

**Integration**: System prompt loads enabled skills as context

### 3.11 Planner (`modules/planner/`)

**Node**: `planner`

**Purpose**: Creates execution plans from goals

### 3.12 Reflection (`modules/reflection/`)

**Node**: `reflection`

**Purpose**: Evaluates agent responses for quality

### 3.13 Annotations (`modules/annotations/`)

**Purpose**: Documentation nodes in flows

---

## 4. Data Flow Analysis

### 4.1 Typical Chat Flow

```
┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐
│  User   │────→│  Chat   │────→│ Memory  │────→│ System  │
│  Input  │     │  Input  │     │  Recall │     │  Prompt │
└─────────┘     └─────────┘     └─────────┘     └────┬────┘
                                                     │
                         ┌───────────────────────────┘
                         ▼
┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐
│  Chat   │←────│  LLM    │←────│Conditional│←───│  Tools  │
│  Output │     │  Core   │     │  Router │     │(if any) │
└─────────┘     └─────────┘     └─────────┘     └─────────┘
     │
     ▼
┌─────────┐
│ Memory  │
│  Save   │
└─────────┘
```

### 4.2 Autonomous Agent Flow

```
┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐
│  User   │────→│  Agent  │────→│  LLM    │────→│  Tool   │
│ Request │     │  Loop   │     │  Call   │     │  Exec   │
└─────────┘     └────┬────┘     └─────────┘     └────┬────┘
                     │                                  │
                     │         ┌────────────────────────┘
                     │         ▼
                     │    ┌─────────┐
                     │    │  Result │────────────────────┐
                     │    │  Store  │                    │
                     │    └─────────┘                    │
                     │         │                         │
                     │         ▼                         │
                     │    ┌─────────┐                    │
                     └────│Reflection│←───────────────────┘
                          │  Check  │
                          └────┬────┘
                               │
                    ┌──────────┴──────────┐
                    ▼                     ▼
              ┌─────────┐           ┌─────────┐
              │Satisfied│           │Not Sat. │
              │  Done   │           │  Retry  │
              └─────────┘           └─────────┘
```

---

## 5. Security Analysis

### 5.1 Tool Sandbox

**Security Layers**:
1. **Code Analysis**: Regex-based dangerous pattern detection
2. **Import Hook**: RestrictedImport class blocks dangerous modules
3. **Restricted Builtins**: Removes eval, exec, open, etc.
4. **File Sandbox**: SafeOpen with directory whitelist
5. **Network Sandbox**: SafeHttpxClient with domain whitelist
6. **Resource Limits**: Timeout, memory, output size
7. **SSRF Protection**: IP range blocking (private networks)

**Vulnerability Mitigations**:
| Threat | Mitigation |
|--------|-----------|
| Code Injection | Static analysis + restricted builtins |
| File Traversal | Path normalization + whitelist |
| Data Exfiltration | Domain whitelist |
| SSRF | IP range blocking |
| Resource Exhaustion | Timeout + memory limits |
| Privilege Escalation | No access to os/sys/subprocess |

### 5.2 Module Isolation

- Each module runs in its own namespace
- Router injection is controlled by ModuleManager
- No direct module-to-module imports required
- Communication through flow data structures

---

## 6. Testing Strategy

### 6.1 Test Coverage

| Category | Count | Key Files |
|----------|-------|-----------|
| Core | 20+ | `test_core_*.py`, `test_flow_runner.py`, `test_flow_manager.py`, `test_flow_integration.py` |
| Memory | 8+ | `test_memory_*.py` |
| Tools | 5+ | `test_tool_*.py`, `test_tools_*.py` |
| Chat | 6+ | `test_chat_module.py`, `test_chat_sessions.py`, `test_chat_features.py`, `test_chat_router_flow.py` |
| Integrations | 7+ | `test_telegram*.py`, `test_calendar*.py`, `test_reasoning_book.py` |
| Security | 3+ | `test_tool_sandbox.py`, `test_sandbox_security.py` |
| E2E | 1 | `test_e2e.py` (requires live server, skipped in CI) |
| **Total** | **926 tests / 67 files** | |

`asyncio_mode = "auto"` is set in `pyproject.toml` — no need for `@pytest.mark.asyncio` decorators. The `conftest.py` auto-backs up `module.json` files during tests to prevent state pollution.

### 6.2 Testing Patterns

**Unit Tests**:
```python
@pytest.mark.asyncio
async def test_node_execution():
    executor = MyNodeExecutor()
    result = await executor.receive({"test": "data"})
    assert result["processed"] is True
```

**Integration Tests**:
```python
def test_flow_execution():
    runner = FlowRunner("test-flow")
    result = asyncio.run(runner.run({"messages": []}))
    assert "content" in result
```

**Security Tests**:
```python
def test_blocks_dangerous_imports():
    with pytest.raises(SecurityError):
        sandbox.execute("import os; os.system('ls')", {})
```

---

## 7. Configuration System

### 7.1 Settings Hierarchy

```
DEFAULT_SETTINGS (code)
    ↓
settings.json (user config)
    ↓
module.json (module config)
    ↓
Node config (flow editor)
    ↓
Input data (runtime)
```

### 7.2 Default Settings

```python
DEFAULT_SETTINGS = {
    "llm_api_url":       "http://localhost:1234/v1",
    "llm_api_key":       "",
    "embedding_api_url": "",
    "default_model":     "local-model",
    "embedding_model":   "",
    "active_ai_flows":   [],
    "temperature":       0.7,
    "max_tokens":        2048,
    "debug_mode":        False,
    "ui_wide_mode":      False,   # Wide layout for the web UI
    "ui_show_footer":    True,    # Show/hide footer in web UI
    "request_timeout":   60.0,
    "max_node_loops":    100,
    "module_allowlist":  [],      # Empty = allow all; non-empty = restrict hot-loading
}
```

### 7.3 Key Configuration Files

| File | Purpose |
|------|---------|
| `settings.json` | Global application settings |
| `ai_flows.json` | Flow definitions |
| `chat_sessions.json` | Chat session history |
| `modules/*/module.json` | Module metadata & config |
| `modules/tools/tools.json` | Tool definitions |
| `data/memory.sqlite3` | Long-term memory relational store |
| `data/memory.faiss` | Long-term memory vector index |
| `data/knowledge_base.sqlite3` | RAG document relational store (FTS5) |
| `data/knowledge_base.faiss` | RAG document vector index |
| `data/reasoning_book.json` | Reasoning journal logs |
| `data/execution_trace.jsonl` | Per-node execution traces (debug) |
| `data/session.json` | Session persistence data |
| `data/episodes/` | EpisodeState JSON files for long-running tasks |

---

## 8. Concurrency Model

### 8.1 Lock Types

| Lock Type | Owner | Usage |
|-----------|-------|-------|
| `threading.RLock` | FlowManager, ModuleManager, SettingsManager, SessionPersistenceManager | Synchronous shared state |
| `threading.Lock` | Metrics, SessionManager (instance), `_init_lock` (singleton guard) | Single-level synchronous guards |
| `asyncio.Lock` | LLMBridge, FlowRunner cache (per event loop), ChatSessions, ReasoningBook | Async resources |

### 8.2 Lock Ordering Rules

1. Never mix threading and asyncio locks in the same code path
2. Always acquire threading.RLock before asyncio.Lock (never the reverse)
3. Lock acquisition order for threading.RLock: `module_manager._lock` → `flow_manager._lock` → `settings._lock`
4. Use context managers (`with`/`async with`)
5. Keep critical sections small — never `await` while holding a threading lock
6. FlowRunner's `_cache_lock` is per-event-loop to avoid "attached to a different loop" errors in tests

See `docs/CONCURRENCY.md` for full details.

---

## 9. Extension Points

### 9.1 Creating New Modules

**Minimum Requirements**:
1. Directory: `modules/my_module/`
2. `module.json` with metadata
3. `__init__.py` (can be empty)

**Optional Components**:
- `router.py` - API routes
- `node.py` - Flow nodes
- `service.py` - Business logic
- `backend.py` - Data layer

### 9.2 Creating New Nodes

**Template**:
```python
class MyNodeExecutor:
    async def receive(self, input_data: dict, config: dict = None) -> dict:
        if input_data is None:
            return None
        # Process data
        return input_data

    async def send(self, processed_data: dict) -> dict:
        return processed_data

async def get_executor_class(node_type_id: str):
    if node_type_id == "my_node":
        return MyNodeExecutor
    return None
```

### 9.3 Creating New Tools

**Requirements**:
1. Add to `modules/tools/tools.json`
2. Create `modules/tools/library/{name}.py`
3. Define `args` parameter and `result` variable

---

## 10. Performance Characteristics

### 10.1 Bottlenecks

| Component | Bottleneck | Mitigation |
|-----------|-----------|------------|
| LLM Calls | Network latency | Async execution |
| Vector Search | FAISS index size | Incremental updates |
| Memory Consolidation | LLM verification | Async background task |
| Tool Execution | Python sandbox | Timeout limits |

### 10.2 Optimizations

- **Executor Caching**: Avoid re-importing node classes
- **FAISS Index**: Batch updates, periodic saves
- **Async I/O**: Non-blocking LLM calls
- **Connection Pooling**: Reuse HTTP clients
- **Lazy Loading**: Modules loaded on demand

---

## 11. Deployment Considerations

### 11.1 Docker Support

**Files**:
- `Dockerfile` - Multi-stage build
- `docker-compose.yml` - Service orchestration

**Volumes**:
- `data/` - Persistent storage
- `modules/` - Custom modules

### 11.2 Environment Variables

| Variable | Purpose |
|----------|---------|
| `LLM_API_URL` | LLM server endpoint |
| `LLM_API_KEY` | API authentication |
| `DEBUG_MODE` | Enable debug logging |
| `TELEGRAM_BOT_TOKEN` | Telegram integration |

---

## 12. Development Workflow

### 12.1 Hot-Reloading

1. Edit module files
2. Disable → Enable module in UI
3. Changes applied without restart

### 12.2 Debugging

- Debug mode: Per-node execution logging
- Flow tracing: Visual execution path
- Error propagation: Structured error messages
- Observability dashboard: Metrics and traces

---

## 13. Code Quality Metrics

### 13.1 Current State

| Metric | Value |
|--------|-------|
| Total Python Files | 165 |
| Lines of Code | ~36,661 |
| Test Files | 67 |
| Test Count | 926 |
| HTML Templates | 33 |
| Modules | 16 |
| Built-in Tools | 23 (16 standard + 7 RLM) |
| Core Classes | 30+ |
| API Endpoints (core routers only) | 40+ |

### 13.2 Patterns Used

| Pattern | Usage |
|---------|-------|
| Async/Await | Throughout |
| Dependency Injection | `core/dependencies.py` |
| Singleton | Settings, Managers |
| Factory | Node executors |
| Strategy | Module loading |
| Observer | Debug logging, Observability |

---

## 14. Recent Security Fixes

All 18 security issues have been fixed:

| Issue | Component | Fix |
|-------|-----------|-----|
| SSRF Vulnerability | FetchURL.py | URL scheme validation + IP range blocking |
| Unsafe eval() | Calculator.py | AST-based SafeEvaluator |
| No TLS Verification | SendEmail.py | ssl.create_default_context() |
| Hardcoded localhost | Goal tools | MEMORY_API_URL environment variable |
| Missing Validation | Goal tools | Added input validation |
| Wrong HTTP Method | SetGoal.py | Changed to JSON |
| Missing Import | Weather.py | Added httpx import |
| Not URL Encoded | Weather.py | Added urllib.parse.quote() |
| No None Guards | ArXivSearch.py | Added null checks |
| No Input Validation | CurrencyConverter.py | Added validation |
| Not Called | WikipediaLookup.py | Added raise_for_status() |
| No Timezone | SystemTime.py | Added timezone info |
| No Length Guard | YouTubeTranscript.py | Added truncation |

---

## 15. Future Enhancement Areas

### 15.1 Scalability

- [ ] Distributed flow execution
- [ ] Horizontal scaling of LLM calls
- [ ] Sharded vector storage

### 15.2 Features

- [ ] Multi-modal support (images, audio)
- [ ] Plugin marketplace
- [ ] Visual debugging tools
- [ ] Flow versioning

### 15.3 Security

- [ ] Code signing for tools
- [ ] Module permission system
- [ ] Audit logging

---

## 16. Conclusion

NeuroCore represents a **well-architected, modular AI orchestration platform** with:

✅ **Clean separation of concerns** between core, modules, and data  
✅ **Extensible plugin system** with hot-swapping support  
✅ **Robust security model** for untrusted code execution  
✅ **Comprehensive memory system** with vector search and consolidation  
✅ **Flexible flow-based programming** for AI agent construction  
✅ **Enterprise-grade observability** with tracing, metrics, and structured logging  
✅ **Scientific schema support** for research applications  
✅ **Production-ready features** including Docker, testing, and monitoring  
✅ **Sophisticated concurrency model** with proper lock ordering  

The codebase demonstrates **mature software engineering practices** with consistent patterns, comprehensive testing, and clear documentation. The modular architecture enables rapid extension while maintaining system stability.

---

*Analysis last updated: 2026-03-14*
*Version: Based on exhaustive codebase review — 165 Python files, 926 tests, 16 modules*

