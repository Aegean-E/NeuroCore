# NeuroCore

<p align="center">
  <img src="https://github.com/Aegean-E/NeuroCore/blob/main/banner.jpg?raw=true" alt="NeuroCore Banner" width="1200">
</p>

<p align="center">
  <a href="https://python.org"><img src="https://img.shields.io/badge/Python-3.12+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"></a>
  <a href="https://fastapi.tiangolo.com"><img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI"></a>
  <a href="https://htmx.org"><img src="https://img.shields.io/badge/HTMX-333333?style=for-the-badge&logo=htmx&logoColor=white" alt="HTMX"></a>
  <a href="https://tailwindcss.com"><img src="https://img.shields.io/badge/Tailwind_CSS-38B2AC?style=for-the-badge&logo=tailwind-css&logoColor=white" alt="Tailwind CSS"></a>
  <a href="https://github.com/Aegean-E/NeuroCore/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License"></a>
</p>

---

**NeuroCore** is a powerful, modular AI agent framework that transforms how you build and deploy autonomous AI applications. Whether you need a smart chatbot with persistent memory, a document-aware assistant, or a fully autonomous agent that can set goals and use tools — NeuroCore provides the complete toolkit.

Built on the principles of **Speed**, **Simplicity**, and **Modularity**, NeuroCore delivers a solid foundation for building custom AI-powered applications with a fast, modern web stack and a powerful visual workflow editor.

---

## 🔥 Why NeuroCore?

- **🎨 Visual AI Flow Editor** — Design complex AI workflows with a drag-and-drop canvas. Chain LLM calls, memory retrieval, knowledge queries, tool execution, and more — all without writing code.

- **🧠 Persistent Long-Term Memory** — Built-in FAISS vector database stores user facts and preferences. Smart extraction and semantic consolidation keep memory organized and relevant.

- **📚 Knowledge Base (RAG)** — Upload PDFs, Markdown, or text files. NeuroCore automatically chunks, embeds, and indexes your documents for intelligent retrieval-augmented generation.

- **🔧 Function Calling & Tools** — Give your AI agency with custom Python tools. From calculators to web search, the LLM can execute code to accomplish real tasks.

- **🤖 Autonomous Agent Capabilities** — Set goals, track progress, and let your agent work independently with the goal system.

- **📱 Multi-Platform** — Built-in Chat UI with multimodal support, Telegram bot integration, and calendar scheduling.

- **⚡ High Performance** — FastAPI backend with HTMX frontend delivers snappy, responsive interactions without heavy JavaScript.

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
Integrated vector database (FAISS + SQLite) for persistent AI memory.

<p align="center">
  <img src="screenshots/memory_browser.png" alt="Memory Browser" width="100%" style="border-radius: 8px; border: 1px solid #334155;">
</p>

- **Automatic Storage** — Background processing saves interactions
- **Smart Extraction** — Uses an Arbiter model to extract facts and preferences
- **Semantic Consolidation** — Intelligent merging of redundant memories
- **Memory Browser** — Dedicated UI to search, filter, edit, merge, and delete memories
- **Conflict Detection** — LLM-powered detection of contradictory memories
- **Meta-Memory Logging** — Full audit trail of all memory operations

### 🧠 Knowledge Base (RAG)
Retrieval-Augmented Generation support for working with documents.

- **Document Ingestion** — Upload PDF, Markdown, or Text files via UI
- **Vector Search** — Documents chunked and embedded into FAISS index
- **Semantic Retrieval** — Knowledge Query node injects relevant context

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

- **Backend**: Python 3.12+, FastAPI, Uvicorn, HTTPX
- **Frontend**: HTMX, TailwindCSS, Vanilla JavaScript
- **Templating**: Jinja2
- **Data**: SQLite, FAISS (vector search)

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
    "embedding_model": ""
}
```

### Running

```bash
python main.py
```

Access at `http://localhost:8000`

---

## 📂 Project Structure

```
NeuroCore/
├── core/                     # Core application logic
│   ├── dependencies.py       # FastAPI dependency injection
│   ├── debug.py              # Debug logging system
│   ├── flow_manager.py       # AI Flow CRUD operations
│   ├── flow_runner.py        # Flow execution engine
│   ├── llm.py                # LLM API client
│   ├── module_manager.py      # Dynamic module loading
│   ├── routers.py            # Main API routes
│   └── settings.py            # Settings management
├── modules/                   # Self-contained feature modules
│   ├── annotations/          # Flow annotation nodes
│   ├── calendar/             # Calendar and event management
│   ├── chat/                 # Chat UI and session management
│   ├── knowledge_base/        # RAG document processing
│   ├── logic/                 # Logic nodes (Delay, Repeater, etc.)
│   ├── llm_module/            # Core LLM node
│   ├── memory/                # Long-term memory (FAISS)
│   ├── memory_browser/       # Memory management UI
│   ├── reasoning_book/       # Reasoning journal
│   ├── system_prompt/          # System prompt injection
│   ├── telegram/              # Telegram bot integration
│   └── tools/                 # Tool library and dispatcher
├── tests/                     # Comprehensive test suite
├── web/
│   └── templates/             # Jinja2 HTML templates
├── main.py                     # FastAPI application entry
├── settings.json              # Runtime configuration
└── ai_flows.json              # Saved AI Flow definitions
```

---

## 🧪 Testing

```bash
# Run all tests
python tests/run_tests.py

# Run with coverage
python tests/run_tests.py --coverage
```

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 📜 License

Licensed under the Apache License 2.0. See LICENSE file for details.

---

<p align="center">
  <strong>NeuroCore</strong> — Build powerful AI agents with modular simplicity
</p>
