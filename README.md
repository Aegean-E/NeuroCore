# NeuroCore

<p align="center">
  <img src="https://github.com/Aegean-E/NeuroCore/blob/main/banner.jpg?raw=true" alt="NeuroCore Banner" width="1200">
</p>

<p align="center">
  <a href="https://python.org"><img src="https://img.shields.io/badge/Python-3.12+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"></a>
  <a href="https://fastapi.tiangolo.com"><img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI"></a>
  <a href="https://htmx.org"><img src="https://img.shields.io/badge/HTMX-333333?style=for-the-badge&logo=htmx&logoColor=white" alt="HTMX"></a>
  <a href="https://tailwindcss.com"><img src="https://img.shields.io/badge/Tailwind_CSS-38B2AC?style=for-the-badge&logo=tailwind-css&logoColor=white" alt="Tailwind CSS"></a>
  <img src="https://img.shields.io/badge/tests-passing-green?style=for-the-badge&color=green" alt="Tests">
  <a href="https://github.com/Aegean-E/NeuroCore/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg?style=for-the-badge" alt="License"></a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/SQLite-003B57?style=for-the-badge&logo=sqlite&logoColor=white" alt="SQLite">
  <img src="https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white" alt="Docker">
  <img src="https://img.shields.io/badge/Jinja2-B41717?style=for-the-badge&logo=jinja&logoColor=white" alt="Jinja2">
  <img src="https://img.shields.io/badge/FAISS-Vector_DB-blueviolet?style=for-the-badge" alt="FAISS">
  <img src="https://img.shields.io/badge/OpenAI_Compatible-412991?style=for-the-badge&logo=openai&logoColor=white" alt="OpenAI Compatible">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/🧠_AI_Memory-gray?style=flat&color=blue" alt="AI Memory">
  <img src="https://img.shields.io/badge/📚_RAG-gray?style=flat&color=green" alt="RAG">
  <img src="https://img.shields.io/badge/🔧_Tools-gray?style=flat&color=orange" alt="Tools">
  <img src="https://img.shields.io/badge/🤖_Autonomous-gray?style=flat&color=purple" alt="Autonomous Agents">
  <img src="https://img.shields.io/badge/🐳_Docker-gray?style=flat&color=cyan" alt="Docker">
  <img src="https://img.shields.io/badge/🔌_Hot--Swap_Modules-gray?style=flat&color=red" alt="Hot-Swap">
  <img src="https://img.shields.io/badge/📊_DAG_Execution-gray?style=flat&color=teal" alt="DAG">
  <img src="https://img.shields.io/badge/🔍_Hybrid_Search-gray?style=flat&color=darkgreen" alt="Hybrid Search">
</p>

---

**NeuroCore** is a powerful, modular AI agent framework that transforms how you build and deploy autonomous AI applications. Whether you need a smart chatbot with persistent memory, a document-aware assistant, or a fully autonomous agent that can set goals and use tools — NeuroCore provides the complete toolkit.

Built on the principles of **Speed**, **Simplicity**, and **Modularity**, NeuroCore delivers a solid foundation for building custom AI-powered applications with a fast, modern web stack and a powerful visual workflow editor.

<p align="center">
  <b>70+ Python files &nbsp;•&nbsp; 40+ HTML templates &nbsp;•&nbsp; 50+ tests &nbsp;•&nbsp; 15 modules &nbsp;•&nbsp; 12 built-in tools &nbsp;•&nbsp; 30+ API routes</b>
</p>

---

## 🔥 Why NeuroCore?

- **🎨 Visual AI Flow Editor** — Design complex AI workflows with a drag-and-drop canvas. Chain LLM calls, memory retrieval, knowledge queries, tool execution, and more — all without writing code.

- **🧠 Persistent Long-Term Memory** — Built-in FAISS vector database stores user facts and preferences. Smart extraction and semantic consolidation keep memory organized and relevant.

- **📚 Knowledge Base (RAG)** — Upload PDFs, Markdown, or text files. NeuroCore automatically chunks, embeds, and indexes your documents for intelligent retrieval-augmented generation.

- **🔧 Function Calling & Tools** — Give your AI agency with custom Python tools. From calculators to web search, the LLM can execute code to accomplish real tasks.

- **🤖 Autonomous Agent Capabilities** — Set goals, track progress, and let your agent work independently with the goal system.

- **📱 Multi-Platform** — Built-in Chat UI with multimodal support, Telegram bot integration, and calendar scheduling.

- **⚡ High Performance** — FastAPI backend with HTMX frontend delivers snappy, responsive interactions without heavy JavaScript.

- **🔌 Hot-Swap Modules** — Enable or disable any module at runtime without restarting the server. Drop a folder into `modules/` to extend the system.

- **🔍 Hybrid Search** — Combines FAISS vector search + SQLite FTS5 keyword search with Reciprocal Rank Fusion (RRF) for best-in-class document retrieval.

---

## 🏗️ Architecture

NeuroCore is organized into **4 clean, decoupled layers**:

```
┌──────────────────────────────────────────────────────────────┐
│           🌐 Web Layer                                        │
│   HTMX + Jinja2 (40+ templates) + TailwindCSS               │
│   Zero heavy JS frameworks — server-driven UI updates        │
├──────────────────────────────────────────────────────────────┤
│           ⚙️  Core Layer                                      │
│   main.py          → FastAPI app + lifespan manager          │
│   core/routers.py  → All HTTP routes (30+ endpoints)         │
│   core/flow_runner.py  → DAG execution engine (Kahn's algo)  │
│   core/flow_manager.py → Flow CRUD (ai_flows.json)           │
│   core/module_manager.py → Dynamic hot-swap module loader    │
│   core/llm.py      → OpenAI-compatible HTTP client           │
│   core/settings.py → Thread-safe settings manager            │
│   core/debug.py    → Structured debug logging system         │
├──────────────────────────────────────────────────────────────┤
│           🔌 Module Layer                                     │
│   modules/<name>/                                            │
│     module.json  → Metadata, config, enabled flag, nodes     │
│     node.py      → Executor classes + dispatcher             │
│     router.py    → FastAPI router (optional)                 │
│     __init__.py  → Exports router for hot-loading            │
├──────────────────────────────────────────────────────────────┤
│           💾 Data Layer                                       │
│   data/memory.sqlite3 + memory.faiss  → Long-term memory     │
│   data/knowledge_base.sqlite3 + .faiss→ RAG documents        │
│   settings.json    → Runtime configuration                   │
│   ai_flows.json    → Saved flow definitions                  │
│   chat_sessions.json → Chat session history                  │
└──────────────────────────────────────────────────────────────┘
```

---

## ✨ Key Features

### 🧠 AI Flow Editor
A visual, node-based canvas to design and orchestrate complex LLM workflows.

<p align="center">
  <img src="screenshots/flow_editor.png" alt="AI Flow Editor Canvas" width="100%" style="border-radius: 8px; border: 1px solid #334155;">
</p>

- **Drag-and-Drop Interface** — Build flows by dragging functions onto the canvas
- **Pan & Zoom** — Effortlessly navigate large and complex flows
- **Multiple Active Flows** — Run multiple flows simultaneously for different tasks
- **Flow Import/Export** — Share flows by exporting as JSON files
- **Singleton Nodes** — Enforce architectural patterns by restricting certain nodes
- **Annotations** — Add comment nodes to document your logic directly on the canvas
- **Keyboard Shortcuts** — Ctrl+A (select all), Ctrl+Z (undo), Delete (remove), Space+Drag (pan), Scroll (zoom)
- **Selection Box** — Click and drag to select multiple nodes
- **Flow Validation** — Pre-execution checks for disabled modules, orphaned connections, and missing tools

### ⚡ Logic & Control Flow
Advanced nodes for complex orchestration:

| Node | Description |
|------|-------------|
| **Delay** | Pause execution for a set duration (configurable in seconds) |
| **Python Scripting** | Execute custom Python code directly within the flow |
| **Repeater** | Create loops or scheduled re-triggers (set to 0 for infinite) |
| **Conditional Router** | Route data based on conditional logic |
| **Scheduled Start** | Wait until a specific date/time before proceeding |
| **Trigger** | Pass-through node for manual triggering |

### 💬 Built-in Chat UI
A clean, modern chat interface for direct interaction with your configured AI flow.

<p align="center">
  <img src="screenshots/chat_ui.png" alt="Chat Interface" width="100%" style="border-radius: 8px; border: 1px solid #334155;">
</p>

- **Multimodal Support** — Upload images to interact with vision-capable models
- **Session Management** — Create, rename, and delete chat sessions
- **Auto-Renaming** — Sessions automatically titled based on conversation context

### 📚 Long-Term Memory
**FAISS + SQLite with LLM-powered filtering — not your average RAG.**

Most AI assistants use naive vector retrieval. NeuroCore's memory system is built different:

<p align="center">
  <img src="screenshots/memory_browser.png" alt="Memory Browser" width="100%" style="border-radius: 8px; border: 1px solid #334155;">
</p>

- **FAISS + SQLite** — Fast vector search backed by persistent storage
- **Arbiter Model** — LLM-powered gate that filters what gets saved (configurable confidence threshold)
- **Semantic Consolidation** — Auto-merges similar/redundant memories to prevent bloat
- **Smart Extraction** — Extracts structured facts (FACT, BELIEF, PREFERENCE, IDENTITY) from conversations
- **Conflict Detection** — LLM identifies contradictory memories
- **TTL & Access Weight** — Old memories fade, frequently accessed ones persist longer
- **Memory Browser** — UI to search, filter, edit, merge, and delete memories
- **Audit Log** — Full `meta_memories` table tracks every edit, merge, delete, and conflict action
- **Goals System** — Dedicated goals table with priority, status, deadline, and context

#### Memory Types

| Type | Description |
|------|-------------|
| `FACT` | Verified, persistent information |
| `BELIEF` | Unverified — expires after 30 days by default |
| `PREFERENCE` | User preferences and tastes |
| `IDENTITY` | Identity facts (name, background, personality) |
| `RULE` | Behavioral rules and instructions |
| `EXPERIENCE` | Past events and experiences |

#### Memory Save Pipeline

```
Conversation Text
      │
      ▼
MemorySaveExecutor.receive()   ← pass-through (non-blocking)
      │
      ▼  asyncio.create_task() — fire and forget
_save_background()
      │
      ▼
LLM Smart Extraction           ← structured JSON facts
      │
      ▼
MemoryArbiter.consider()       ← confidence threshold gate
      │
      ▼
MemoryConsolidator             ← auto-merge similar memories (every N hours)
      │
      ▼
MemoryStore.add_entry()        ← FAISS index + SQLite
```

#### Memory Recall Pipeline

```
User Message → Embedding → FAISS Search → Score Filter
      │
      ▼
_memory_context injected into input_data
      │
      ▼
SystemPromptExecutor           ← picks up _memory_context
      │
      ▼
Injected into system message   ← LLM receives relevant memories
```

### 🧠 Knowledge Base (RAG)
Retrieval-Augmented Generation with **hybrid search** for working with documents.

- **Document Ingestion** — Upload PDF, Markdown, or Text files via UI
- **Vector Search** — Documents chunked and embedded into FAISS index
- **Keyword Search** — SQLite FTS5 full-text search with auto-sync triggers
- **Hybrid Search** — Reciprocal Rank Fusion (RRF, k=60) combines vector + keyword results
- **Semantic Retrieval** — Knowledge Query node injects relevant context
- **Self-Healing Index** — Automatically rebuilds FAISS index on startup if out of sync
- **Integrity Checks** — Detects chunk count mismatches and missing embeddings

#### Knowledge Base Search Modes

| Mode | Implementation | Best For |
|------|---------------|----------|
| **Vector Search** | FAISS `IndexFlatIP` + L2 normalization | Semantic similarity |
| **Keyword Search** | SQLite FTS5 virtual table | Exact term matching |
| **Hybrid Search** | Reciprocal Rank Fusion (RRF) | Best overall accuracy |

### 📅 Calendar & Scheduling
Manage time-sensitive tasks and events.

- **Visual Calendar** — Full GUI to view and manage events
- **Event Watcher** — Flow node that checks for upcoming events
- **Scheduled Execution** — Trigger actions at specific times

### 🛠️ Tools Library
Define and manage custom Python functions that the LLM can execute.

<p align="center">
  <img src="screenshots/tools_library.png" alt="Tool Library Editor" width="100%" style="border-radius: 8px; border: 1px solid #334155;">
</p>

- **Function Calling** — Full support for OpenAI-compatible function calling
- **Visual Editor** — Create tools with JSON schema validation
- **Hot-Reloading** — Tools saved as Python files, loaded dynamically
- **Tool Dispatcher** — Execute tools requested by the LLM
- **Import/Export** — Share tools as JSON or Python files
- **Per-Node Control** — `allowed_tools` config restricts which tools each dispatcher can use
- **Rate Limiting** — `max_tools_per_turn` (default: 5) prevents runaway tool loops

#### Built-in Tools

| Tool | Description |
|------|-------------|
| **Calculator** | Evaluates mathematical expressions |
| **ConversionCalculator** | Converts units (temperature, length, weight, volume) |
| **CurrencyConverter** | Real-time currency conversion (Frankfurter API) |
| **TimeZoneConverter** | Timezone conversions (IANA format) |
| **Weather** | Current weather for any location |
| **SystemTime** | Current date and time |
| **FetchURL** | Extracts text content from URLs |
| **SendEmail** | Sends emails via SMTP |
| **SaveReminder** | Saves calendar events/reminders |
| **CheckCalendar** | Retrieves upcoming events |
| **YouTubeTranscript** | Fetches YouTube video transcripts |
| **WikipediaLookup** | Searches Wikipedia articles |
| **ArXivSearch** | Searches academic papers |
| **SetGoal** | Creates a new goal for autonomous agents |
| **MarkGoalComplete** | Marks a goal as completed |
| **DeleteGoal** | Deletes a goal |
| **Check Goal** | Flow node to inject goal into context |

### 📱 Telegram Integration
Connect your AI flow to Telegram for remote access.

- **Chat Remotely** — Interact with your AI from anywhere
- **Vision Support** — Send photos to analyze with vision models
- **Command Control** — Manage sessions with `/new_session`, `/delete_session`

### 📖 Reasoning Book
A reasoning journal for AI agents.

- **Thought Recording** — Store reasoning steps during flow execution
- **Context Injection** — Load previous reasoning into LLM context

### 🔌 Modular Architecture
NeuroCore is built around a powerful, plugin-based architecture.

- **Self-Contained Modules** — Each feature is an isolated package
- **Hot-Swapping** — Enable/disable modules at runtime without restart
- **Easy Extensibility** — Drop a folder into `modules/` with a `module.json`
- **Thread-Safe** — All module state mutations protected by `threading.Lock()`
- **Config Persistence** — Module configs written back to `module.json` automatically

#### Available Modules

| Module | Purpose | Router | Flow Node |
|--------|---------|--------|-----------|
| `chat` | Chat UI + session management | ✅ | ✅ |
| `memory` | FAISS+SQLite long-term memory | ✅ | ✅ |
| `knowledge_base` | RAG document ingestion | ✅ | ✅ |
| `tools` | Tool library + dispatcher | ✅ | ✅ |
| `system_prompt` | System prompt injection | ✅ | ✅ |
| `llm_module` | Core LLM call node | ✅ | ✅ |
| `telegram` | Telegram bot integration | ✅ | ✅ |
| `calendar` | Calendar + event management | ✅ | ✅ |
| `reasoning_book` | Reasoning journal | ✅ | ✅ |
| `memory_browser` | Memory management UI | ✅ | — |
| `logic` | Delay, Repeater, Conditional, etc. | — | ✅ |
| `annotations` | Flow comment nodes | — | ✅ |
| `planner` | Planner node | — | ✅ |
| `agent_loop` | Agent loop node | — | ✅ |
| `reflection` | Reflection node | — | ✅ |

---

## ⚙️ How It Works — The Flow Engine

The `FlowRunner` executes AI flows as a **Directed Acyclic Graph (DAG)** using Kahn's topological sort algorithm.

### Node Execution Contract

Every node executor implements two async methods:

```python
async def receive(self, input_data: dict, config: dict = None) -> dict | None:
    # Process input. Return None to STOP the branch (conditional logic).
    ...

async def send(self, processed_data: dict) -> dict:
    # Return output passed to all downstream nodes.
    ...
```

### Key Engine Mechanisms

| Mechanism | Implementation |
|-----------|---------------|
| **Topological Sort** | Kahn's algorithm for deterministic execution ordering |
| **Cycle Detection** | Heuristic break — picks lowest in-degree node to continue |
| **Bridge Nodes** | Implicit parallel connections (BFS component grouping) |
| **Conditional Routing** | `_route_targets` key in output dict for branching |
| **Context Propagation** | `messages` key auto-preserved across all nodes |
| **Loop Guard** | `max_node_loops` setting (default: 1,000) |
| **Executor Cache** | Class-level cache avoids re-importing modules per execution |
| **Dynamic Import** | `importlib.import_module()` + `reload()` for hot code updates |
| **Background Tasks** | `asyncio.create_task()` tracked in `app.state.background_tasks` |
| **Auto-Start** | Repeater nodes with incoming connections start automatically on app launch |

### Typical Flow Execution

```
Chat Input  ──►  Memory Recall  ──►  System Prompt  ──►  LLM Core
                                                              │
                                                    ┌─────────┴──────────┐
                                                    ▼                    ▼
                                            Tool Dispatcher        Chat Output
                                                    │
                                                    ▼
                                              LLM Core (2nd pass)
                                                    │
                                                    ▼
                                              Chat Output
```

---

## 🧩 Available AI Flow Nodes

### Input Nodes
| Node | Description |
|------|-------------|
| **Chat Input** | Receives user messages from the chat interface |
| **Telegram Input** | Receives messages from Telegram |

### Processing Nodes
| Node | Description |
|------|-------------|
| **LLM Core** | Calls the configured LLM with messages |
| **System Prompt** | Injects system prompts and enables tools |
| **Memory Save** | Saves content to long-term memory |
| **Memory Recall** | Retrieves relevant memories semantically |
| **Knowledge Query** | Queries the knowledge base for context |
| **Check Goal** | Injects current goal into context |
| **Reasoning Load** | Loads reasoning history into context |

### Output Nodes
| Node | Description |
|------|-------------|
| **Chat Output** | Sends responses to the chat interface |
| **Telegram Output** | Sends responses to Telegram |
| **Tool Dispatcher** | Executes tools requested by the LLM |

### Logic Nodes
| Node | Description |
|------|-------------|
| **Trigger** | Pass-through node for manual triggering |
| **Delay** | Pauses execution for specified seconds |
| **Python Script** | Executes custom Python code |
| **Repeater** | Re-triggers flow after delay (supports loops) |
| **Conditional Router** | Routes based on field existence |
| **Scheduled Start** | Waits until specific date/time |

### Utility Nodes
| Node | Description |
|------|-------------|
| **Annotation** | Adds comments to document flow logic |

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **Backend** | Python 3.12+, FastAPI, Uvicorn, HTTPX |
| **Frontend** | HTMX, TailwindCSS, Vanilla JavaScript |
| **Templating** | Jinja2 (40+ templates) |
| **Vector Database** | FAISS (`faiss-cpu`) — `IndexIDMap(IndexFlatIP)` |
| **Relational Database** | SQLite (WAL mode, FTS5 full-text search) |
| **LLM Integration** | OpenAI-compatible API (Ollama, LM Studio, LocalAI, OpenAI) |
| **Testing** | pytest, pytest-asyncio, pytest-httpx, pytest-cov |
| **Deployment** | Docker + docker-compose |

### Runtime Dependencies

```
fastapi  •  uvicorn  •  httpx  •  jinja2  •  numpy  •  faiss-cpu  •  python-multipart
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.12 or higher
- An OpenAI-compatible LLM API endpoint (Ollama, LM Studio, LocalAI, etc.)

### Installation

```bash
# Clone the repository
git clone https://github.com/Aegean-E/NeuroCore
cd NeuroCore

# Create virtual environment
python -m venv venv

# Activate (Windows)
.\venv\Scripts\activate
# Or (macOS/Linux)
source venv/bin/activate

# Install dependencies
pip install fastapi uvicorn httpx jinja2 numpy faiss-cpu "pytest<9" "pytest-cov" "pytest-httpx" "pytest-asyncio" python-multipart
```

### Configuration

The application uses `settings.json`. On first run, it will be created with defaults. Update `llm_api_url` to point to your LLM:

```json
{
    "llm_api_url": "http://localhost:1234/v1",
    "llm_api_key": "",
    "default_model": "local-model",
    "embedding_api_url": "",
    "embedding_model": "",
    "temperature": 0.7,
    "max_tokens": 2048,
    "debug_mode": true,
    "request_timeout": 60.0,
    "max_node_loops": 1000
}
```

### Running

```bash
python main.py
```

Access at `http://localhost:8000`

### Docker

NeuroCore can also be run using Docker:

```bash
# Build and run
docker-compose up -d --build

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

Access at `http://localhost:8000`

**Note:** Create a `settings.json` with your LLM configuration before running, or it will be created with defaults on first start. The `data/` folder and `settings.json` are mounted from your host for persistence.

---

## 📂 Project Structure

```
NeuroCore/
├── core/                       # Core application logic
│   ├── dependencies.py         # FastAPI dependency injection
│   ├── debug.py                # Debug logging system
│   ├── flow_manager.py         # AI Flow CRUD operations
│   ├── flow_runner.py          # Flow execution engine (DAG)
│   ├── llm.py                  # LLM API client (OpenAI-compatible)
│   ├── module_manager.py       # Dynamic module loading & hot-swap
│   ├── routers.py              # Main API routes (30+ endpoints)
│   └── settings.py             # Thread-safe settings manager
├── modules/                    # Self-contained feature modules (15)
│   ├── agent_loop/             # Agent loop node
│   ├── annotations/            # Flow annotation nodes
│   ├── calendar/               # Calendar and event management
│   ├── chat/                   # Chat UI and session management
│   ├── knowledge_base/         # RAG document processing (hybrid search)
│   ├── logic/                  # Logic nodes (Delay, Repeater, etc.)
│   ├── llm_module/             # Core LLM node
│   ├── memory/                 # Long-term memory (FAISS + SQLite)
│   ├── memory_browser/         # Memory management UI
│   ├── planner/                # Planner node
│   ├── reasoning_book/         # Reasoning journal
│   ├── reflection/             # Reflection node
│   ├── system_prompt/          # System prompt injection
│   ├── telegram/               # Telegram bot integration
│   └── tools/                  # Tool library and dispatcher
│       └── library/            # Built-in tool Python files (12)
├── tests/                      # Comprehensive test suite (50+ files)
├── web/
│   └── templates/              # Jinja2 HTML templates (40+)
├── data/                       # Persistent data (SQLite + FAISS indexes)
├── screenshots/                # UI screenshots
├── main.py                     # FastAPI application entry point
├── pyproject.toml              # Project metadata & build config
├── requirements.txt            # Python dependencies
├── settings.json               # Runtime configuration
├── ai_flows.json               # Saved AI Flow definitions
├── Dockerfile                  # Container build instructions
└── docker-compose.yml          # Container orchestration
```

---

## 🌐 API Reference

NeuroCore exposes a comprehensive REST API:

| Group | Key Endpoints |
|-------|--------------|
| **Dashboard** | `GET /` · `GET /dashboard/gui` · `GET /dashboard/stats` · `GET /dashboard/recent-sessions` |
| **Modules** | `GET /modules/list` · `GET /modules/{id}/details` · `POST /modules/{id}/config` · `POST /modules/{id}/enable` · `POST /modules/{id}/disable` · `POST /modules/reorder` |
| **AI Flow** | `GET /ai-flow` · `POST /ai-flow/save` · `GET /ai-flow/{id}` · `GET /ai-flow/{id}/validate` · `POST /ai-flow/{id}/set-active` · `POST /ai-flow/{id}/run-node/{node_id}` · `POST /ai-flow/{id}/delete` |
| **Settings** | `GET /settings` · `POST /settings/save` · `POST /settings/reset` · `GET /settings/export/config` · `POST /settings/import/config` · `GET /settings/export/flows` · `POST /settings/import/flows` |
| **Debug** | `GET /debug` · `GET /debug/logs` · `GET /debug/events` · `POST /debug/clear` |
| **System** | `GET /llm-status` · `GET /navbar` · `GET /footer` · `GET /system-time` |

---

## 🧪 Testing

Comprehensive test suite with **50+ test files** covering all layers:

```bash
# Run all tests
python tests/run_tests.py

# Run with coverage
python tests/run_tests.py --coverage
```

### Coverage Areas

| Area | Test Files |
|------|-----------|
| **Core Engine** | `test_flow_runner.py`, `test_flow_manager.py`, `test_flow_integration.py`, `test_flow_validation.py` |
| **Module System** | `test_module_manager.py`, `test_dependencies.py` |
| **Memory** | `test_memory_nodes.py`, `test_memory_arbiter.py`, `test_memory_consolidation.py`, `test_memory_router.py`, `test_memory_browser.py` |
| **Knowledge Base** | `test_knowledge_base.py`, `test_knowledge_base_improvements.py` |
| **Chat** | `test_chat_module.py`, `test_chat_sessions.py`, `test_chat_features.py`, `test_chat_router_flow.py` |
| **Tools** | `test_tools_library.py` |
| **LLM** | `test_core_llm.py`, `test_llm_node.py` |
| **Robustness** | `test_core_concurrency.py`, `test_core_robustness.py`, `test_core_improvements.py` |
| **Integrations** | `test_telegram_module.py`, `test_calendar.py`, `test_reasoning_book.py`
