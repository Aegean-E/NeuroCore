# NeuroCore

<p align="center">
  <img src="https://github.com/Aegean-E/NeuroCore/blob/main/banner.jpg?raw=true" alt="NeuroCore Banner" width="1200">
</p>

<p align="center">
  <a href="https://python.org"><img src="https://img.shields.io/badge/Python-3.12+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"></a>
  <a href="https://fastapi.tiangolo.com"><img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI"></a>
  <a href="https://htmx.org"><img src="https://img.shields.io/badge/HTMX-333333?style=for-the-badge&logo=htmx&logoColor=white" alt="HTMX"></a>
  <a href="https://tailwindcss.com"><img src="https://img.shields.io/badge/Tailwind_CSS-38B2AC?style=for-the-badge&logo=tailwind-css&logoColor=white" alt="Tailwind CSS"></a>
  <img src="https://img.shields.io/badge/tests-1141_passing-brightgreen?style=for-the-badge" alt="Tests">
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
  <img src="https://img.shields.io/badge/🧠_Persistent_Memory-4f46e5?style=flat-square" alt="AI Memory">
  <img src="https://img.shields.io/badge/📚_RAG_Hybrid_Search-16a34a?style=flat-square" alt="RAG">
  <img src="https://img.shields.io/badge/🔧_23_Built--in_Tools-ea580c?style=flat-square" alt="Tools">
  <img src="https://img.shields.io/badge/🤖_Autonomous_Agents-7c3aed?style=flat-square" alt="Autonomous Agents">
  <img src="https://img.shields.io/badge/🌐_Community_Marketplace-0891b2?style=flat-square" alt="Marketplace">
  <img src="https://img.shields.io/badge/🔌_Hot--Swap_Modules-dc2626?style=flat-square" alt="Hot-Swap">
  <img src="https://img.shields.io/badge/📊_Visual_Flow_Editor-0d9488?style=flat-square" alt="DAG">
  <img src="https://img.shields.io/badge/🔒_Secure_Sandbox-d97706?style=flat-square" alt="Secure Sandbox">
</p>

<h3 align="center">
  A complete, self-hosted AI agent framework.<br>
  Visual flow editor · Persistent memory · RAG · Tools · Multi-platform messaging · Community marketplace.
</h3>

<p align="center">
  <a href="#-quick-start">Quick Start</a> •
  <a href="#%EF%B8%8F-architecture">Architecture</a> •
  <a href="#-key-features">Features</a> •
  <a href="#-available-modules">Modules</a> •
  <a href="#-api-reference">API</a> •
  <a href="#-testing">Testing</a>
</p>

---

> **NeuroCore** is a production-quality, modular AI orchestration platform. Wire LLM calls, memory retrieval, document search, tool execution, and autonomous agent loops into visual DAG workflows — then deploy with a single command. Everything runs locally, works with any OpenAI-compatible model, and ships with 18 hot-swappable modules out of the box.

<p align="center">
  <b>18 modules &nbsp;·&nbsp; 28 flow nodes &nbsp;·&nbsp; 23 built-in tools &nbsp;·&nbsp; 39 UI templates &nbsp;·&nbsp; 74 core API routes &nbsp;·&nbsp; 1,141 tests</b>
</p>

---

## 🚀 Quick Start

```bash
# 1. Clone & install
git clone https://github.com/Aegean-E/NeuroCore
cd NeuroCore
pip install -r requirements.txt

# 2. Point to your LLM (edit settings.json)
# "llm_api_url": "http://localhost:1234/v1"   ← Ollama, LM Studio, LocalAI, OpenAI…

# 3. Run
py main.py          # Windows
python main.py      # macOS / Linux

# 4. Open http://localhost:8000
```

Or with Docker:

```bash
docker-compose up -d --build
# → http://localhost:8000
```

**`settings.json` key fields:**
```json
{
    "llm_api_url":      "http://localhost:1234/v1",
    "llm_api_key":      "",
    "default_model":    "local-model",
    "embedding_api_url": "",
    "embedding_model":  "",
    "temperature":      0.7,
    "max_tokens":       2048,
    "debug_mode":       false,
    "module_allowlist": []
}
```

> `module_allowlist` — restrict hot-loadable modules (empty = allow all).
> `debug_mode` — enables per-node execution traces in `data/execution_trace.jsonl`.

---

## 🏗️ Architecture

NeuroCore is organized into **four clean, decoupled layers:**

```
┌────────────────────────────────────────────────────────────────┐
│  🌐  Presentation Layer                                        │
│  HTMX + Jinja2 (39 templates) + TailwindCSS                   │
│  Chat UI · Flow Editor (Canvas) · Memory Browser · Dashboard  │
├────────────────────────────────────────────────────────────────┤
│  ⚙️  Core Layer                                                │
│  flow_runner.py  → DAG engine (Kahn's sort, bridge groups)     │
│  flow_manager.py → Flow CRUD + version history (20 ver/flow)   │
│  module_manager.py → Hot-swap module loader                    │
│  routers.py      → 74 HTTP endpoints (incl. full marketplace)  │
│  llm.py          → OpenAI-compatible async client              │
│  observability.py → Distributed tracing · metrics · logging    │
│  session_manager.py → Chat sessions + EpisodeState persist.    │
│  settings.py     → Thread-safe config (RLock, atomic write)    │
├────────────────────────────────────────────────────────────────┤
│  🔌  Module Layer  (18 modules · 28 node executors)            │
│  modules/<name>/                                               │
│    module.json  → Metadata, config, enabled flag               │
│    node.py      → Executor classes + dispatcher                │
│    router.py    → FastAPI router (optional)                    │
├────────────────────────────────────────────────────────────────┤
│  💾  Data Layer                                                │
│  SQLite (WAL + FTS5) · FAISS IndexFlatIP · JSON · JSONL        │
│  memory.sqlite3 + memory.faiss  → Long-term memory            │
│  knowledge_base.sqlite3 + .faiss → RAG documents              │
│  data/marketplace/  → Community catalog + uploads             │
│  data/episodes/     → EpisodeState (long-running tasks)        │
└────────────────────────────────────────────────────────────────┘
```

### Flow Engine

Every workflow is a **Directed Acyclic Graph (DAG)** executed by `FlowRunner`:

```
Chat Input ──► Memory Recall ──► System Prompt ──► LLM Core
                                                       │
                                          ┌────────────┴────────────┐
                                          ▼                         ▼
                                   Tool Dispatcher             Chat Output
                                          │
                                          ▼
                                   LLM Core (2nd pass)
                                          │
                                          ▼
                                     Chat Output
```

| Mechanism | How |
|-----------|-----|
| **Topological Sort** | Kahn's algorithm — deterministic, dependency-respecting order |
| **Bridge Groups** | BFS component grouping for implicit parallel data sharing |
| **Conditional Routing** | `_route_targets` key — branch at runtime without extra wires |
| **Loop Guard** | `max_node_loops` counter (default 100, max 1,000) |
| **Executor Cache** | Class-level FIFO cache (max 100) — no re-imports per execution |
| **Episode Persistence** | `EpisodeState` in `data/episodes/` — resume long-running tasks |
| **Timeout** | `asyncio.wait_for` wraps every flow run |
| **Input Isolation** | Deep-copy of `messages` before each node — no cross-node mutation |

---

## ✨ Key Features

### 🎨 Visual AI Flow Editor

<p align="center">
  <img src="screenshots/flow_editor.png" alt="AI Flow Editor Canvas" width="100%" style="border-radius: 8px; border: 1px solid #334155;">
</p>

Build complex AI pipelines without writing orchestration code.

| Capability | Detail |
|------------|--------|
| Drag-and-drop canvas | Add, move, and wire nodes visually |
| Pan & zoom | Navigate flows of any size |
| Multiple active flows | Run several flows in parallel |
| Flow version history | Up to 20 saved versions per flow with one-click restore |
| Import / Export | Share flows as JSON files |
| Singleton nodes | Enforce architectural constraints |
| Annotation nodes | Document logic directly on the canvas |
| Keyboard shortcuts | `Ctrl+A` select all · `Ctrl+Z` undo · `Space+drag` pan · Scroll zoom |
| Pre-flight validation | Checks for disabled modules, orphaned connections, missing tools |

---

### 🧠 Persistent Long-Term Memory

<p align="center">
  <img src="screenshots/memory_browser.png" alt="Memory Browser" width="100%" style="border-radius: 8px; border: 1px solid #334155;">
</p>

Not just vector search — a full memory lifecycle with LLM-powered quality control.

```
Conversation → LLM Extraction → Arbiter (confidence gate) → Consolidator → FAISS + SQLite
```

| Feature | Detail |
|---------|--------|
| **Dual storage** | FAISS `IndexFlatIP` + SQLite WAL for fast vector + structured queries |
| **Arbiter** | LLM-powered gate filters low-confidence memories before saving |
| **Auto-consolidation** | Merges semantically similar memories every 24 h (cosine > 0.92 threshold) |
| **Conflict detection** | LLM identifies and flags contradictory memories |
| **TTL decay** | BELIEFs expire after 30 days; frequently accessed memories persist longer |
| **Memory Browser** | Full UI to search, filter, edit, merge, and delete memories |
| **Audit log** | `meta_memories` table records every edit, merge, delete, and conflict |
| **Goals system** | Dedicated goals table with priority, deadline, and status |

**Memory types:** `FACT` · `BELIEF` (30-day TTL) · `PREFERENCE` · `IDENTITY` · `RULE` · `EXPERIENCE`

---

### 📚 Knowledge Base (RAG)

Upload documents, get intelligent retrieval.

| Feature | Detail |
|---------|--------|
| **Ingestion** | PDF, Markdown, plain text via drag-and-drop UI |
| **Hybrid search** | FAISS vector search + SQLite FTS5 combined with **Reciprocal Rank Fusion (RRF, k=60)** |
| **Incremental indexing** | Only re-indexes changed documents — provenance tracking via timestamps |
| **Self-healing** | Rebuilds FAISS index on startup if chunk counts are mismatched |
| **Flow integration** | `query_knowledge` node injects retrieved context into `_kb_context` |

| Search Mode | Implementation | Best for |
|-------------|---------------|----------|
| Vector | FAISS `IndexFlatIP` + L2 norm | Semantic similarity |
| Keyword | SQLite FTS5 virtual table | Exact term matching |
| **Hybrid** | Reciprocal Rank Fusion | Best overall accuracy |

---

### 🔧 Tools Library & Secure Sandbox

<p align="center">
  <img src="screenshots/tools_library.png" alt="Tool Library Editor" width="100%" style="border-radius: 8px; border: 1px solid #334155;">
</p>

Give your AI real capabilities with custom Python tools — executed **safely**.

**23 built-in tools across 6 categories:**

<details>
<summary><b>🧮 Calculations & Conversions (5 tools)</b></summary>

| Tool | Description |
|------|-------------|
| `Calculator` | Evaluates math expressions (AST-based, no `eval`) |
| `ConversionCalculator` | Converts units — temperature, length, weight, volume |
| `CurrencyConverter` | Real-time conversion via Frankfurter API |
| `TimeZoneConverter` | Converts between IANA timezones |
| `SystemTime` | Returns current date and time |

</details>

<details>
<summary><b>🌐 Web & Search (5 tools)</b></summary>

| Tool | Description |
|------|-------------|
| `Weather` | Current weather for any location |
| `FetchURL` | Extracts text content from a URL (SSRF-protected) |
| `WikipediaLookup` | Search and retrieve Wikipedia articles |
| `ArXivSearch` | Search academic papers by keyword |
| `YouTubeTranscript` | Fetch full transcripts from YouTube videos |

</details>

<details>
<summary><b>📅 Calendar & Goals (5 tools)</b></summary>

| Tool | Description |
|------|-------------|
| `SaveReminder` | Create calendar events and reminders |
| `CheckCalendar` | Retrieve upcoming events |
| `SetGoal` | Create a new goal for autonomous agents |
| `MarkGoalComplete` | Mark a goal as completed |
| `DeleteGoal` | Delete a goal |

</details>

<details>
<summary><b>📧 Communication (1 tool)</b></summary>

| Tool | Description |
|------|-------------|
| `SendEmail` | Send emails via SMTP (TLS verified) |

</details>

<details>
<summary><b>🧠 RLM Tools — Recursive Language Model (7 tools)</b></summary>

For agents working on long or complex inputs that exceed context limits.

| Tool | Description |
|------|-------------|
| `Peek` | View a character-range slice of the prompt |
| `Search` | Find regex matches in the current prompt |
| `Chunk` | Split prompt into manageable overlapping chunks |
| `SubCall` | Recursively invoke an LLM on a sub-prompt |
| `SetVariable` | Store intermediate results by name |
| `GetVariable` | Retrieve a stored result |
| `SetFinal` | Set the final answer and terminate processing |

</details>

**Sandbox security — five independent layers:**

| Layer | Mechanism |
|-------|-----------|
| Static analysis | Scans code for dangerous patterns before execution |
| Import blocklist | Blocks `sys`, `subprocess`, `socket`, `pickle`, `ctypes`, `pathlib`, and 15+ more |
| Module mocking | `import httpx` → `SafeHttpxClient`; `import os` → `SafeEnv` (env vars only) |
| Network guard | Domain allowlist + SSRF protection (blocks all private IP ranges) |
| Resource limits | 30 s timeout · 100 KB max output · optional memory cap |

---

### 🤖 Autonomous Agent Loop

Let the agent think, use tools, and iterate — without human input.

```
[repeater_node] (every N seconds)
      │
      ▼
[memory_recall] → [system_prompt] → [agent_loop]
                                          │
                               ┌──────────┴──────────┐
                               │  LLM ↔ Tool loop    │
                               │  (up to N iterations)│
                               └──────────┬──────────┘
                                          │
                                   [messaging_output]
                           (proactive_recipients: ["telegram:123"])
```

The `agent_loop` node runs an autonomous LLM↔tool execution loop until the model stops calling tools or `max_iterations` is reached — no external trigger needed.

**Configuration:**
```json
{
    "max_iterations": 10,
    "tool_error_strategy": "continue",
    "timeout": 120,
    "include_memory_context": true,
    "include_knowledge_context": true,
    "include_plan_in_context": true
}
```

---

### 🌐 Community Marketplace

Share and discover AI flows, skills, tools, and prompts.

| Feature | Detail |
|---------|--------|
| **Immutable identity** | HMAC-SHA256 handle (12-char hex) — tamper-proof author identity per instance |
| **Item types** | Flows (`.json`), Skills (`.md`), Tools (`.json`/`.py`), Prompts |
| **One-click import** | Tools → registered in `tools.json` + written to `library/`; Flows → imported into `ai_flows.json` |
| **Versioning** | Update notes prepended to `changelog` on each publish; "Update Available" shown for outdated imports |
| **Comments** | Threaded comments with `@handle` autocomplete, delete-own-comment, custom confirm modal |
| **Notifications** | Bell icon + unread badge; triggered by comments on your items and `@mention`s |
| **Voting** | Upvote/downvote per item with score-based sorting |
| **Originality** | Backend rejects re-upload of unmodified imported items |
| **Community profiles** | Public uploader pages with bio, item grid, stats; "Preview as visitor" mode |

---

### 📱 Multi-Platform Messaging

Connect your AI flow to messaging platforms via the unified `messaging_bridge` module.

| Platform | Mechanism | Message limit |
|----------|-----------|--------------|
| **Telegram** | HTTP long-polling | 3,072 chars/chunk |
| **Discord** | WebSocket Gateway v10 | 1,900 chars/chunk |
| **Signal** | HTTP polling (signal-cli REST) | 1,800 chars/chunk |
| **WhatsApp** | Webhook (Evolution API) | 4,000 chars/chunk |

Use `messaging_output` with `proactive_recipients` to push messages autonomously — no incoming trigger required.

### 📧 Email Bridge

The `email_bridge` module connects flows to email.

- **IMAP receive** — polls inbox and injects messages as flow input
- **SMTP send** — sends replies or proactive emails via authenticated SMTP

---

### 💬 Chat UI

<p align="center">
  <img src="screenshots/chat_ui.png" alt="Chat Interface" width="100%" style="border-radius: 8px; border: 1px solid #334155;">
</p>

| Feature | Detail |
|---------|--------|
| Real-time streaming | LLM tokens streamed token-by-token via SSE |
| Thinking trace | Agent reasoning steps displayed live in the UI |
| Multimodal | Upload images for vision-capable models |
| Session management | Create, rename, delete chat sessions |
| Auto-rename | Sessions titled automatically from conversation context |
| Session compaction | LLM summarizes old context; keeps last N turns verbatim — prevents token overflow |

---

### 📋 Skills Management

Manage reusable instruction files that inject into system prompts.

- **SKILL.md files** — create best practices, personas, and domain-specific guidelines
- **In-UI editor** — create and edit skill files directly from Settings
- **Import / Export** — share skills as Markdown files
- **Marketplace integration** — publish and import skills from the community

---

### 📊 Observability

Production-grade monitoring baked in — no external services needed.

- **Distributed tracing** — span-based traces with `trace_id`/`span_id` propagation via `contextvars`
- **Metrics** — counters, gauges, histograms with p50/p95/p99 percentiles; persisted across restarts
- **Structured logging** — JSON-formatted logs with trace context correlation
- **Debug mode** — `debug_mode: true` writes per-node execution traces to `data/execution_trace.jsonl`

---

## 🧩 Available Modules

NeuroCore ships **18 hot-swappable modules**. Enable or disable any of them at runtime — no restart required.

| Module | Purpose | Router | Flow Nodes |
|--------|---------|:------:|:----------:|
| `llm_module` | Core LLM node — streaming, tool calling, vision | ✅ | ✅ |
| `agent_loop` | Autonomous LLM↔tool loop (3 nodes) | — | ✅ |
| `system_prompt` | System prompt injection + tool registration | ✅ | ✅ |
| `memory` | FAISS+SQLite long-term memory (3 nodes) | ✅ | ✅ |
| `knowledge_base` | RAG — hybrid FTS5+FAISS search | ✅ | ✅ |
| `tools` | Tool library, dispatcher, secure sandbox | ✅ | ✅ |
| `logic` | Control flow — Delay, Repeater, Conditional, Script, Schedule, ContextLengthRouter | — | ✅ (7) |
| `chat` | Chat UI + SSE streaming + session compaction | ✅ | ✅ |
| `messaging_bridge` | Telegram / Discord / Signal / WhatsApp | ✅ | ✅ |
| `email_bridge` | IMAP receive + SMTP send | ✅ | ✅ |
| `planner` | Goal decomposition into executable steps | — | ✅ |
| `reflection` | Response quality gate (`satisfied` → conditional routing) | — | ✅ |
| `calendar` | Event scheduling and calendar watcher node | ✅ | ✅ |
| `skills` | Instruction file management | ✅ | — |
| `memory_browser` | Full UI to search, edit, merge, delete memories | ✅ | — |
| `reasoning_book` | Thought journal — save and load reasoning context | ✅ | ✅ |
| `browser_automation` | Lazy Playwright singleton (headless browser) | ✅ | — |
| `annotations` | Comment nodes for documenting flows | — | ✅ |

> **Adding a module:** drop a folder into `modules/` with a `module.json` and `__init__.py`. It appears in the dashboard immediately. See [`docs/MODULE_GUIDE.md`](docs/MODULE_GUIDE.md).

---

## 🧩 Available AI Flow Nodes

**28 built-in node executors** across all modules:

### Input Nodes
| Node | Module | Description |
|------|--------|-------------|
| `chat_input` | chat | Receives user messages from the web UI |
| `messaging_input` | messaging_bridge | Receives messages from any configured platform |
| `email_input` | email_bridge | Polls IMAP inbox for new messages |

### Processing Nodes
| Node | Module | Description |
|------|--------|-------------|
| `llm_module` | llm_module | Calls the LLM — streaming, tool calling, vision |
| `system_prompt` | system_prompt | Injects system prompt, merges memory/kb/reasoning context |
| `memory_recall` | memory | Semantic FAISS vector search → `_memory_context` |
| `memory_save` | memory | Async background extraction + arbiter + consolidation |
| `check_goal` | memory | Injects highest-priority active goal into context |
| `query_knowledge` | knowledge_base | Hybrid RAG search → `_kb_context` |
| `tool_dispatcher` | tools | Executes LLM-requested tool calls in the sandbox |
| `agent_loop` | agent_loop | Autonomous LLM↔tool loop with configurable max iterations |
| `recursive_lm` | agent_loop | RLM node for long-context recursive processing |
| `repl_environment` | agent_loop | REPL-style code execution environment |
| `planner` | planner | Decomposes a goal into an ordered step list |
| `plan_step_tracker` | planner | Tracks step completion and routes to next step |
| `reflection` | reflection | LLM evaluates output quality → `satisfied` bool |
| `reasoning_save` | reasoning_book | Persists reasoning steps to journal |
| `reasoning_load` | reasoning_book | Injects reasoning history into context |
| `calendar_watcher` | calendar | Checks for upcoming calendar events |

### Output Nodes
| Node | Module | Description |
|------|--------|-------------|
| `chat_output` | chat | Sends response to the web UI |
| `messaging_output` | messaging_bridge | Routes reply to originating platform or proactive recipients |
| `email_output` | email_bridge | Sends email via SMTP |

### Logic & Control Nodes
| Node | Module | Description |
|------|--------|-------------|
| `trigger_node` | logic | Pass-through manual trigger |
| `delay_node` | logic | Pause execution for N seconds |
| `script_node` | logic | Run custom Python code (sandboxed) |
| `repeater_node` | logic | Re-trigger flow on a timer (0 = infinite) |
| `conditional_router` | logic | Branch on field existence (`tool_calls`, `satisfied`, etc.) |
| `schedule_start_node` | logic | Wait until a specific ISO 8601 datetime |
| `context_length_router` | logic | Route to RLM vs standard LLM based on token estimate |
| `comment_node` | annotations | Resizable text note on the canvas |

### Reserved Flow Keys

| Key | Owner | Purpose |
|-----|-------|---------|
| `messages` | All nodes | Conversation history — preserved across all nodes |
| `_memory_context` | `memory_recall` | Injected memory context |
| `_kb_context` | `query_knowledge` | Knowledge base retrieval results |
| `_route_targets` | Conditional router | Dynamic branch targets (consumed by FlowRunner) |
| `tool_calls` | `llm_module` | LLM-requested tool invocations |
| `tool_results` | `tool_dispatcher` | Tool execution results |
| `requires_continuation` | `tool_dispatcher` | Multi-turn tool loop flag |
| `satisfied` | `reflection` | Boolean for conditional routing |
| `_is_error` | Internal | Marks the payload as a flow error |

---

## 🛠️ Tech Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| **Backend** | Python 3.12+, FastAPI 0.115+, Uvicorn 0.32+ | Async-first, fast, typed |
| **Frontend** | HTMX + TailwindCSS + Vanilla JS | No build step, server owns state |
| **Templating** | Jinja2 3.1+ (39 templates) | Server-rendered HTML fragments for HTMX |
| **Vector DB** | FAISS `IndexFlatIP` + L2 normalization | Sub-millisecond similarity search |
| **Relational DB** | SQLite WAL mode + FTS5 | ACID compliance + full-text search, zero infra |
| **HTTP Client** | HTTPX 0.28+ (async, connection pooling) | Non-blocking LLM calls |
| **WebSocket** | websockets 12.0+ | Discord Gateway, custom protocols |
| **LLM API** | OpenAI-compatible | Works with Ollama, LM Studio, LocalAI, OpenAI, etc. |
| **Validation** | Pydantic 2.10+ | Schema enforcement for flow payloads and structured output |
| **Testing** | pytest + pytest-asyncio + pytest-httpx + pytest-cov | 1,141 tests across 72 files |
| **Deployment** | Docker + docker-compose | Single-command production deploy |
| **Linting** | Ruff | Fast Python linter + formatter |

**Runtime dependencies:**
```
fastapi · uvicorn · httpx · jinja2 · numpy · faiss-cpu · python-multipart · filelock
```

---

## 📂 Project Structure

<details>
<summary><b>Click to expand full tree</b></summary>

```
NeuroCore/
├── core/                         # Core framework (15 files)
│   ├── flow_runner.py            # DAG execution engine (Kahn's sort, bridge groups, episodes)
│   ├── flow_manager.py           # Flow CRUD + version history (up to 20 versions/flow)
│   ├── module_manager.py         # Hot-swap module loader (_loaded_once safety)
│   ├── routers.py                # 74 HTTP endpoints + full marketplace implementation
│   ├── llm.py                    # Async OpenAI-compatible LLM client (connection pooling)
│   ├── settings.py               # Thread-safe config manager (RLock + atomic write)
│   ├── observability.py          # Distributed tracing · metrics · structured JSON logging
│   ├── session_manager.py        # Chat sessions + EpisodeState persistence
│   ├── structured_output.py      # Pydantic-based structured output with retry logic
│   ├── planner_helpers.py        # Plan dependency graphs + cycle detection
│   ├── flow_context.py           # FlowContext Pydantic model
│   ├── flow_data.py              # FlowData TypedDict + helper functions
│   ├── errors.py                 # 14 typed exception classes
│   ├── debug.py                  # Structured debug logging
│   ├── dependencies.py           # FastAPI dependency injection
│   └── schemas/                  # Scientific domain models (Hypothesis, Article, Finding, StudyDesign)
├── modules/                      # 18 self-contained feature modules
│   ├── agent_loop/               # Autonomous agent loop (3 nodes)
│   ├── annotations/              # Flow comment nodes
│   ├── browser_automation/       # Lazy Playwright singleton (headless=true)
│   ├── calendar/                 # Calendar events + watcher node
│   ├── chat/                     # Chat UI + SSE streaming + session compaction
│   ├── email_bridge/             # IMAP polling + SMTP sending
│   ├── knowledge_base/           # RAG — hybrid FTS5+FAISS with RRF
│   ├── llm_module/               # Core LLM node — streaming, tool calling, vision
│   ├── logic/                    # 7 control-flow nodes
│   ├── memory/                   # Long-term memory — FAISS+SQLite+arbiter+consolidation
│   ├── memory_browser/           # Memory management UI
│   ├── messaging_bridge/         # Telegram/Discord/Signal/WhatsApp
│   ├── planner/                  # Goal decomposition (planner + plan_step_tracker)
│   ├── reasoning_book/           # Thought journal (save + load nodes)
│   ├── reflection/               # Quality gate node
│   ├── skills/                   # Instruction file management
│   ├── system_prompt/            # System prompt injection + tool registration
│   └── tools/                    # Tool library (23 tools) + dispatcher + sandbox
│       ├── library/              # 16 standard tool implementations
│       └── rlm_library/          # 7 RLM tool implementations
├── web/templates/                # 39 Jinja2 HTML templates
├── tests/                        # 72 test files · 1,141 tests
│   └── run_tests.py              # Test runner with optional coverage
├── data/                         # Runtime data (mutable, excluded from hot-reloader)
│   ├── memory.sqlite3            # Long-term memory relational store
│   ├── memory.faiss              # Long-term memory vector index
│   ├── knowledge_base.sqlite3    # RAG relational store (FTS5)
│   ├── knowledge_base.faiss      # RAG vector index
│   ├── reasoning_book.json       # Thought journal
│   ├── execution_trace.jsonl     # Per-node execution traces (debug_mode only)
│   ├── episodes/                 # EpisodeState files for long-running agent tasks
│   ├── marketplace/              # Community catalog (catalog.json) + uploads/
│   ├── marketplace_profile.json  # Local uploader identity (handle, username, bio)
│   ├── marketplace_notifications.json  # Notification queue (capped at 200)
│   └── download_history.json     # Import dedup tracking
├── docs/                         # Deep-dive documentation
│   ├── SYSTEM_ARCHITECTURE.md    # Architecture reference
│   ├── PROJECT_ANALYSIS.md       # Full codebase analysis
│   ├── MODULE_GUIDE.md           # How to build modules
│   ├── TOOL_GUIDE.md             # How to build tools + sandbox reference
│   ├── CONCURRENCY.md            # Lock ordering rules + deadlock prevention
│   └── IDEAS.md                  # Feature backlog
├── main.py                       # FastAPI app entry point + lifespan manager
├── settings.json                 # Runtime configuration
├── ai_flows.json                 # Saved flow definitions
├── ai_flows_versions.json        # Flow version history
├── pyproject.toml                # Project metadata · pytest config · ruff rules
├── requirements.txt              # Runtime dependencies
├── Dockerfile
└── docker-compose.yml
```

</details>

---

## 🌐 API Reference

74 routes in `core/routers.py` alone. Key groups:

| Group | Endpoints |
|-------|-----------|
| **Dashboard** | `GET /` · `GET /dashboard/stats` · `GET /dashboard/recent-sessions` |
| **AI Flows** | `GET /ai-flow` · `POST /ai-flow/save` · `GET /ai-flow/{id}/validate` · `POST /ai-flow/{id}/run-node/{nid}` · `POST /ai-flow/{id}/delete` · *(version history, rename, set-active, make-default…)* |
| **Modules** | `GET /modules/list` · `POST /modules/{id}/config` · `POST /modules/{id}/{action}` · `POST /modules/reorder` |
| **Settings** | `GET /settings` · `POST /settings/save` · `POST /settings/reset` · `GET/POST /settings/export|import` |
| **Marketplace** | `GET /marketplace` · `GET /marketplace/item/{id}` · `POST /marketplace/upload` · `POST /marketplace/item/{id}/import` · `POST /marketplace/item/{id}/vote` · `POST /marketplace/item/{id}/update` · `DELETE /marketplace/item/{id}` · `GET /marketplace/uploader/{handle}` · `POST /marketplace/item/{id}/comment` · `DELETE /marketplace/item/{id}/comment/{cid}` · `GET /marketplace/notifications` · `POST /marketplace/notifications/read` · `DELETE /marketplace/notifications` · `GET /marketplace/handles` · `GET/POST /marketplace/profile` |
| **Debug** | `GET /debug/logs` · `GET /debug/events` · `GET /debug/summary` · `GET /debug/agent-summary` · `POST /debug/clear` |
| **System** | `GET /llm-status` · `GET /navbar` · `GET /footer` · `GET /system-time` · `GET /goals` |

Each module also exposes its own router (e.g. `/chat/*`, `/memory/*`, `/knowledge-base/*`, `/tools/*`).

---

## 🧪 Testing

```bash
# Run all tests
py tests/run_tests.py

# With coverage report
py tests/run_tests.py --coverage

# Single file
pytest tests/test_tool_sandbox.py -v

# E2E tests (requires live server)
NEUROCORE_RUN_E2E=1 pytest tests/test_e2e.py -v
```

`asyncio_mode = "auto"` is set globally — **never add `@pytest.mark.asyncio`.**
`conftest.py` auto-backs up and restores `module.json` files around each test session.

**Coverage breakdown:**

| Area | Files | Tests |
|------|------:|------:|
| Core flow engine | 6 | 120+ |
| Chat + sessions | 6 | 90+ |
| Agent loop | 3 | 60+ |
| Memory system | 6 | 100+ |
| Tools + sandbox | 3 | 80+ |
| Individual modules | 14 | 220+ |
| LLM bridge | 4 | 60+ |
| Concurrency / robustness | 2 | 40+ |
| Messaging bridge | 1 | 63 |
| E2E (live server) | 1 | gated |
| **Total** | **72** | **1,141+** |

**Security testing — sandbox enforcement:**
```python
# Dangerous imports are blocked
def test_blocks_dangerous_modules():
    with pytest.raises(SecurityError):
        execute_sandboxed("import os; result = os.getcwd()", {})

# SSRF protection
def test_blocks_internal_ips():
    assert SafeHttpxClient()._is_ip_blocked('127.0.0.1') == True
```

---

## 🤝 Contributing

Contributions are welcome. See:
- [`docs/MODULE_GUIDE.md`](docs/MODULE_GUIDE.md) — build a new module in minutes
- [`docs/TOOL_GUIDE.md`](docs/TOOL_GUIDE.md) — add tools with sandbox-aware Python
- [`docs/SYSTEM_ARCHITECTURE.md`](docs/SYSTEM_ARCHITECTURE.md) — deep architecture reference
- [`docs/CONCURRENCY.md`](docs/CONCURRENCY.md) — lock ordering rules (read before touching shared state)

```bash
# Run tests before submitting
pytest tests/ -x      # stop on first failure
ruff check .          # lint
ruff format .         # format
```

**Ideas for contributions:**
- New messaging platform bridges (Slack, Matrix, etc.)
- Browser automation flow nodes (Playwright executor)
- New domain-specific tools (finance, science, DevOps)
- Flow visual replay from `execution_trace.jsonl`
- Dashboard analytics from `core/observability.py` metrics

---

## 📄 License

NeuroCore is licensed under the [Apache 2.0 License](LICENSE).

---

## 🙏 Acknowledgments

- **[FAISS](https://github.com/facebookresearch/faiss)** — Facebook AI Similarity Search
- **[FastAPI](https://fastapi.tiangolo.com)** — Modern async Python web framework
- **[HTMX](https://htmx.org)** — HTML over the wire
- **[Ollama](https://ollama.ai) / [LM Studio](https://lmstudio.ai) / [LocalAI](https://localai.io)** — Making local LLMs accessible

---

<p align="center">
  <b>Built with ❤️ for the AI community</b><br><br>
  <a href="https://github.com/Aegean-E/NeuroCore">⭐ Star us on GitHub</a>
</p>
