# NeuroCore Module Development Guide

This guide explains how to create new modules for NeuroCore.

## 1. Directory Structure

Create a new folder inside `modules/` with your module ID (e.g., `my_new_module`).

```text
modules/
└── my_new_module/
    ├── __init__.py       # Exposes the router (if any)
    ├── module.json       # Metadata (Required)
    ├── router.py         # FastAPI routes (Optional)
    └── node.py           # AI Flow logic (Optional)
```

## 2. Module Metadata (`module.json`)

Create a `module.json` file. This defines how NeuroCore loads and displays your module.

```json
{
    "name": "My New Module",
    "description": "Description of what this module does.",
    "enabled": true,
    "id": "my_new_module",
    "is_flow_node": false,  // Set to true if the module itself acts as a single node
    "order": 10,            // Sort order in the UI
    "provides_nodes": [     // List of nodes this module adds to the AI Flow editor
        {
            "id": "my_custom_node",
            "name": "Custom Processor",
            "description": "Processes text in a specific way.",
            "singleton": false, // If true, only one instance allowed per flow
            "configurable": true // If true, shows a settings gear in the UI
        }
    ]
}
```

## 3. Adding API Routes (`router.py`)

If your module needs a backend API or UI pages, create `router.py`.

```python
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

router = APIRouter()
templates = Jinja2Templates(directory="web/templates")

@router.get("/gui", response_class=HTMLResponse)
async def my_module_gui(request: Request):
    return "<div>Hello from My Module!</div>"

@router.post("/do-something")
async def do_something():
    return {"status": "success"}
```

**Important:** You must expose this router in `__init__.py`:

```python
# modules/my_new_module/__init__.py
from .router import router
```

## 4. Adding AI Flow Nodes (`node.py`)

If your module provides nodes for the AI Flow editor, create `node.py`.

You need to define an **Executor** class for each node type and a `get_executor_class` dispatcher.

```python
class MyCustomNodeExecutor:
    async def receive(self, input_data: dict, config: dict = None) -> dict:
        """
        Receives data from the previous node.
        input_data usually contains keys like 'messages', 'content', etc.
        """
        # Process data
        result = input_data.copy()
        result['processed'] = True
        return result

    async def send(self, processed_data: dict) -> dict:
        """
        Sends data to the next node.
        """
        return processed_data

async def get_executor_class(node_type_id: str):
    """
    Dispatcher function called by FlowRunner.
    Returns the class (not instance) for the given node type ID.
    """
    if node_type_id == 'my_custom_node':
        return MyCustomNodeExecutor
    return None
```

## 5. Hot-Swapping

NeuroCore supports hot-loading.
1. Create your folder and files.
2. Go to the **Modules** page in the UI.
3. Your module should appear. Enable it to load the router.