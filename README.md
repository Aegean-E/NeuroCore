# NeuroCore

<p align="center">
  <img src="https://github.com/Aegean-E/NeuroCore/blob/main/banner.jpg?raw=true" alt="NeuroCore Banner" width="1200">
</p>


**A modular, extensible, and high-performance web UI for interacting with and orchestrating local or remote Language Models.**

NeuroCore is built on the principles of **Speed**, **Simplicity**, and **Modularity**. It provides a solid foundation for building custom AI-powered applications with a fast, modern web stack and a powerful visual workflow editor.

---

## 🚀 Core Philosophy

*   **Speed**: The backend is built with **FastAPI**, one of the fastest Python web frameworks available. The frontend uses **HTMX** to deliver fast, server-rendered partials, avoiding a heavy client-side JavaScript footprint.
*   **Simplicity**: The project maintains a clean, logical structure. Data is stored in simple JSON files for easy setup and inspection. The frontend logic is co-located with the HTML, making it easy to understand and maintain.
*   **Modularity**: The entire system is designed around a powerful module manager. Features can be enabled, disabled, and created as self-contained packages. This "hot-swapping" capability allows for dynamic extension of the core application without requiring a server restart.

## ✨ Key Features

*   **🔌 Modular Architecture**: Easily extend the application by dropping new modules into the `modules/` directory. Enable or disable them on the fly from the UI.
*   **🧠 AI Flow Editor**: A visual, node-based canvas to design and orchestrate complex LLM workflows.
    *   **Drag-and-Drop Interface**: Build flows by dragging functions onto the canvas.
    *   **Pan & Zoom**: Effortlessly navigate large and complex flows.
    *   **Singleton Nodes**: Enforce architectural patterns by restricting certain nodes (like Chat Input/Output) to a single instance per flow.
    *   **Flow Management**: Create, save, rename, and switch between multiple AI flows to handle different tasks.
*   **⚡ Logic & Control Flow**: Advanced nodes for complex orchestration.
    *   **Delay**: Pause execution for a set duration.
    *   **Python Scripting**: Execute custom Python code directly within the flow to transform data.
    *   **Repeater**: Create loops or scheduled re-triggers of AI flows.
    *   **Conditional Router**: Branch flows based on data conditions, tool execution results, or custom logic.
*   **💬 Built-in Chat UI**: A clean, modern chat interface for direct interaction with your configured AI flow.
    *   **Multimodal Support**: Upload images to interact with vision-capable models.
    *   **Session Management**: Create, rename, and delete chat sessions to organize your conversations.
    *   **Auto-Renaming**: Sessions are automatically titled based on the conversation context.
*   **📚 Long-Term Memory**: Integrated vector database (FAISS + SQLite) for persistent AI memory.
    *   **Automatic Storage**: Background processing saves user and assistant interactions.
    *   **Smart Extraction**: Uses an Arbiter model to extract specific facts and preferences, filtering out noise.
    *   **Semantic Consolidation**: Intelligent merging of redundant memories to prevent database bloat and maintain coherence.
    *   **Memory Browser**: A dedicated UI to search, filter, and delete stored memories.
    *   **Context Injection**: Automatically retrieves relevant memories during conversations.
*   **🛠️ Tools Library**: Define and manage custom Python functions (tools) that the LLM can execute.
    *   **Function Calling**: Full support for OpenAI-compatible function calling.
    *   **Visual Editor**: Create and edit tools directly in the UI with JSON schema validation.
    *   **Hot-Reloading**: Tools are saved as Python files and loaded dynamically.
    *   **Tool Dispatcher**: A dedicated flow node to execute tools requested by the LLM.
*   **📱 Telegram Integration**: Connect your AI flow to Telegram for remote access.
    *   **Chat Remotely**: Interact with your AI agent from anywhere via the Telegram app.
    *   **Vision Support**: Send photos to Telegram to analyze images using vision-capable models.
    *   **Command Control**: Manage sessions (`/new_session`, `/delete_session`) directly from the chat.
*   **📝 Annotations**: Add visual comments and documentation directly onto the AI Flow canvas using Comment nodes.
*   **⚙️ Dynamic Configuration**: Manage LLM API endpoints, models, and other parameters through a simple, tabbed settings UI.
*   **✅ Robust Testing**: A comprehensive test suite using `pytest` to ensure code quality and stability.

## 🛠️ Tech Stack

*   **Backend**:
    *   Python 3.12+
    *   FastAPI: For the high-performance, asynchronous API.
    *   Uvicorn: As the ASGI server.
    *   HTTPX: For making async requests to the LLM API.
*   **Frontend**:
    *   HTMX: For modern, dynamic browser behavior directly from HTML.
    *   Tailwind CSS: For the utility-first styling.
    *   **Vanilla JavaScript**: Used exclusively for the interactive AI Flow canvas.
*   **Templating**:
    *   Jinja2: For server-side HTML templating.

## ⚙️ Getting Started

Follow these steps to get NeuroCore up and running on your local machine.

### 1. Prerequisites

*   **Python 3.12 or higher.**
*   An **OpenAI-compatible LLM API endpoint**. This can be from a local server like LM Studio, Ollama, or any other service that exposes a `/v1/chat/completions` endpoint.

### 2. Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd NeuroCore
    ```

2.  **Create and activate a virtual environment** (recommended):
    ```bash
    # For Windows
    python -m venv venv
    .\venv\Scripts\activate

    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required dependencies:**
    *(You may need to create a `requirements.txt` file based on the libraries used, such as `fastapi`, `uvicorn`, `httpx`, `jinja2`, `pytest`, etc.)*
    ```bash
    pip install fastapi uvicorn httpx jinja2 numpy faiss-cpu "pytest<9" "pytest-cov" "pytest-httpx" "pytest-asyncio"
    ```

### 3. Configuration

1.  The application uses `settings.json` for configuration. If it doesn't exist, it will be created with default values on the first run.
2.  Run the application once to generate the file, or create it manually.
3.  Open `settings.json` and update the `llm_api_url` to point to your running LLM service.

    ```json
    {
        "llm_api_url": "http://localhost:1234/v1",
        "llm_api_key": "",
        "default_model": "local-model",
        "embedding_api_url": "",
        "embedding_model": "",
        "active_ai_flow": "default-flow-001"
    }
    ```

### 4. Running the Application

Execute the `main.py` file to start the web server:

```bash
python main.py
```

The application will be available at `http://localhost:8000`.

## 📂 Project Structure

```
NeuroCore/
├── core/                 # Core application logic (managers, dependencies)
├── modules/              # Self-contained, plug-and-play feature modules
│   ├── chat/             # The chat UI and flow nodes
│   ├── logic/            # Scripting, delays, and flow control
│   └── llm_module/       # The core LLM flow node
│   ├── memory/           # Long-term memory backend and nodes
│   ├── memory_browser/   # UI for managing memories
│   └── system_prompt/    # System prompt injection node
│   ├── tools/            # Tool library and dispatcher
│   ├── telegram/         # Telegram bot integration
│   └── annotations/      # Flow documentation and comments
├── tests/                # The pytest test suite
├── web/                  # Frontend files
│   ├── static/           # Static assets (future CSS/JS)
│   └── templates/        # Jinja2 HTML templates
├── ai_flows.json         # Stores saved AI Flows
├── chat_sessions.json    # Stores chat histories
├── main.py               # FastAPI application entry point
├── settings.json         # System and model configuration
└── README.md             # This file
```

## 🧩 Creating a New Module

The modular architecture is NeuroCore's strongest feature. Here’s how to create your own module.

### Step 1: Create the Module Directory

Create a new folder inside the `modules/` directory. For example, `modules/my_new_module`.

### Step 2: Define the Module Metadata

Create a `module.json` file inside your new folder. This file tells NeuroCore about your module.

```json
{
    "name": "My New Module",
    "description": "A short description of what this module does.",
    "enabled": false,
    "id": "my_new_module",
    "is_flow_node": true,  // Is this module itself a single node?
    "singleton": false,    // Can only one instance of this node exist in a flow?
    "order": 99,           // Determines display order in the UI
    "provides_nodes": [    // A list of nodes this module provides to the AI Flow
        {
            "id": "my_custom_node",
            "name": "My Custom Node",
            "description": "A custom processing step.",
            "singleton": false
        }
    ]
}
```

### Step 3: Add Functionality

#### To Add API Routes (like the Chat module):

1.  Create a `router.py` file in your module directory to define your FastAPI `APIRouter`.
2.  Create an `__init__.py` file in the same directory and expose your router:
    ```python
    # modules/my_new_module/__init__.py
    from .router import router
    ```

#### To Add AI Flow Nodes:

1.  Create a `node.py` file in your module directory.
2.  Define one or more "Executor" classes. Each class must have `async def receive(self, data)` and `async def send(self, data)` methods.
3.  Create a dispatcher function `get_executor_class(node_type_id)` that returns the correct class based on the `id` from your `module.json`.

    ```python
    # modules/my_new_module/node.py
    class MyCustomNodeExecutor:
        async def receive(self, input_data: dict) -> dict:
            # Process the data...
            return input_data

        async def send(self, processed_data: dict) -> dict:
            # Pass the result to the next node
            return processed_data

    async def get_executor_class(node_type_id: str):
        if node_type_id == 'my_custom_node':
            return MyCustomNodeExecutor
        return None
    ```

## 🔧 Tool Library

NeuroCore includes a powerful **Tool Library** that implements OpenAI-compatible function calling. This allows your AI agents to interact with external APIs, databases, or perform calculations.

1.  **Define**: Create tools in the **Tools** tab using a visual editor. You define the JSON schema for parameters and the Python code to execute.
2.  **Enable**: In your AI Flow, select the **System Prompt** node and enable the specific tools you want the agent to use.
3.  **Execute**: Add a **Tool Dispatcher** node to your flow. When the LLM decides to call a function, the dispatcher executes your Python code and returns the result to the LLM.

For a comprehensive guide on creating and using tools, see modules/TOOL_GUIDE.md.

## 🧪 Testing

The project includes a robust test suite using `pytest`. To run the tests, execute the provided runner script:

```bash
python tests/run_tests.py
```

To include a coverage report, use the `--coverage` flag:

```bash
python tests/run_tests.py --coverage
```

## 📜 License

This project is licensed under the **Apache License, Version 2.0**. See the LICENSE file for full details.

---
