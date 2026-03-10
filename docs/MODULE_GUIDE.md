# 🧩 NeuroCore Module Development Guide

> **Build powerful extensions for NeuroCore with custom modules, nodes, and APIs**

---

## 📑 Table of Contents

1. [Quick Start](#1-quick-start)
2. [Architecture Overview](#2-architecture-overview)
3. [Directory Structure](#3-directory-structure)
4. [Module Metadata](#4-module-metadata)
5. [Adding API Routes](#5-adding-api-routes)
6. [Adding AI Flow Nodes](#6-adding-ai-flow-nodes)
7. [Core Modules Reference](#7-core-modules-reference)
8. [Node Implementation Patterns](#8-node-implementation-patterns)
9. [Advanced Topics](#9-advanced-topics)
10. [Hot-Swapping & Development](#10-hot-swapping--development)
11. [Troubleshooting](#11-troubleshooting)

---

## 1. Quick Start

Create a new module in 3 simple steps:

```bash
# 1. Create module directory
mkdir modules/my_module

# 2. Create metadata file
cat > modules/my_module/module.json << 'EOF'
{
    "name": "My Module",
    "description": "What this module does",
    "enabled": true,
    "id": "my_module",
    "order": 10
}
EOF

# 3. Create empty init file
touch modules/my_module/__init__.py
```

🎉 Your module now appears in the NeuroCore dashboard!

---

## 2. Architecture Overview

NeuroCore's modular architecture separates concerns into three layers:

```
┌─────────────────────────────────────────────────────────────┐
│                    AI Flow Layer                            │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐        │
│  │  Nodes  │──│  Nodes  │──│  Nodes  │──│  Nodes  │        │
│  │(Logic)  │  │(Memory) │  │(Tools)  │  │(Chat)   │        │
│  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘        │
└───────┼────────────┼────────────┼────────────┼─────────────┘
        │            │            │            │
        └────────────┴────────────┴────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   Module Manager Layer                      │
│         (Loads modules, manages lifecycle)                  │
└─────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI Router Layer                     │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐        │
│  │  GUI    │  │  API    │  │  WebSocket│  │  Static │        │
│  │ Routes  │  │ Endpoints│  │  Handlers│  │  Files  │        │
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘        │
└─────────────────────────────────────────────────────────────┘
```

### Module Types

| Type | Purpose | Example |
|------|---------|---------|
| **Flow Node** | Single node for AI Flows | LLM Core, System Prompt |
| **Node Provider** | Multiple nodes for flows | Logic & Flow, Memory |
| **Service Module** | Backend services + UI | Knowledge Base, Calendar |
| **Bridge Module** | External integrations | Telegram Bridge |

---

## 3. Directory Structure

Create a new folder inside `modules/` with your module ID:

```
modules/
└── my_new_module/
    ├── __init__.py          # Exposes the router (required)
    ├── module.json          # Metadata (required)
    ├── router.py            # FastAPI routes (optional)
    ├── node.py              # AI Flow logic (optional)
    ├── service.py           # Business logic (optional)
    └── backend.py           # Data layer (optional)
```

### File Purposes

| File | Required | Purpose |
|------|----------|---------|
| `module.json` | ✅ Yes | Module metadata and configuration |
| `__init__.py` | ✅ Yes | Package initialization, router exposure |
| `router.py` | ❌ No | FastAPI routes for UI and API endpoints |
| `node.py` | ❌ No | AI Flow node implementations |
| `service.py` | ❌ No | Business logic layer |
| `backend.py` | ❌ No | Data persistence layer |

---

## 4. Module Metadata

The `module.json` file defines how NeuroCore loads and displays your module.

### Complete Schema

```json
{
    "name": "My New Module",
    "description": "Description of what this module does.",
    "enabled": true,
    "id": "my_new_module",
    "icon": "M12 2L2 7l10 5 10-5-10-5z...",
    
    "is_flow_node": false,
    "singleton": false,
    "order": 10,
    
    "config": {
        "my_setting": "default_value",
        "api_key": "",
        "timeout": 30
    },
    
    "provides_nodes": [
        {
            "id": "my_custom_node",
            "name": "Custom Processor",
            "description": "Processes text in a specific way.",
            "singleton": false,
            "configurable": true,
            "config": {
                "param1": "value1"
            }
        }
    ],
    
    "load_error": null
}
```

### Field Reference

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Display name in UI |
| `description` | string | Short description for tooltips |
| `enabled` | boolean | Whether module is active |
| `id` | string | Unique identifier (snake_case) |
| `icon` | string | SVG path data for sidebar icon |
| `is_flow_node` | boolean | Module acts as single flow node |
| `singleton` | boolean | Only one instance allowed |
| `order` | number | Sort order in UI (lower = first) |
| `config` | object | Default configuration values |
| `provides_nodes` | array | Nodes provided for AI Flows. Note: The `config` field in `provides_nodes` is for documentation/display purposes only. Per-node runtime configuration is defined in the node's executor class in `node.py` and exposed via the node's config UI in the flow editor. |
| `load_error` | string/null | System tracking field |

### Icon Format

Icons use SVG path data. Extract from SVG files:

```svg
<svg viewBox="0 0 24 24">
  <path d="M12 2L2 7l10 5 10-5-10-5z"/>
</svg>
```

Use the `d` attribute value in `module.json`.

---

## 5. Adding API Routes

Create `router.py` for backend APIs and UI pages.

### Basic Router Structure

```python
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

router = APIRouter()
templates = Jinja2Templates(directory="web/templates")

@router.get("/gui", response_class=HTMLResponse)
async def module_gui(request: Request):
    """Returns the module's GUI fragment."""
    return templates.TemplateResponse(
        request, 
        "my_module_gui.html", 
        {"data": "value"}
    )

@router.get("/api/data")
async def get_data():
    """API endpoint for module data."""
    return {"status": "success", "data": []}

@router.post("/api/action")
async def perform_action(request: Request):
    """Perform an action."""
    data = await request.json()
    return {"result": "done"}
```

### Exposing the Router

**Required** in `__init__.py`:

```python
# modules/my_new_module/__init__.py
from .router import router

# Optional: Import other components
from .node import get_executor_class
from .service import MyService
```

### Router Patterns

| Pattern | Use Case | Example |
|---------|----------|---------|
| `/gui` | HTML fragment for UI | Knowledge base interface |
| `/api/*` | REST API endpoints | CRUD operations |
| `/ws` | WebSocket handlers | Real-time updates |
| `/webhook` | External callbacks | Telegram bot webhooks |

---

## 6. Adding AI Flow Nodes

Create `node.py` to provide nodes for the AI Flow editor.

### Node Executor Pattern

```python
class MyNodeExecutor:
    """Executor class for a flow node."""
    
    async def receive(self, input_data: dict, config: dict = None) -> dict:
        """
        Process incoming data from previous node.
        
        Args:
            input_data: Data from previous node (messages, content, etc.)
            config: Node configuration + system metadata (_flow_id, _node_id)
            
        Returns:
            Processed data for next node
        """
        if input_data is None:
            return None
            
        config = config or {}
        
        # Process data
        result = input_data.copy()
        result['processed'] = True
        result['my_field'] = config.get('setting', 'default')
        
        return result

    async def send(self, processed_data: dict) -> dict:
        """
        Send data to next node.
        
        Args:
            processed_data: Data from receive() method
            
        Returns:
            Final output for next node
        """
        return processed_data


async def get_executor_class(node_type_id: str):
    """
    Dispatcher function called by FlowRunner.
    
    Args:
        node_type_id: The node type identifier from module.json
        
    Returns:
        Executor class or None
    """
    if node_type_id == 'my_custom_node':
        return MyNodeExecutor
    return None
```

### Input/Output Data Structure

Common keys in `input_data`:

| Key | Type | Description |
|-----|------|-------------|
| `messages` | list | Conversation history |
| `content` | string | Direct text content |
| `choices` | list | LLM response format |
| `_memory_context` | string | Injected memory context |
| `knowledge_context` | string | Injected knowledge context |
| `reasoning_context` | string | Injected reasoning context |
| `_flow_id` | string | Current flow ID (in config) |
| `_node_id` | string | Current node ID (in config) |
| `_repeat_count` | int | Repeater iteration count |

---

## 7. Core Modules Reference

NeuroCore includes **16 built-in modules**:

### 🧠 Core AI Modules

#### LLM Core (`llm_module`)
Direct interface to Large Language Models.

| Property | Value |
|----------|-------|
| **Type** | Flow Node |
| **Singleton** | Yes |
| **Order** | 0 |

**Configuration:**
```json
{
    "temperature": 0.7,
    "max_tokens": 8192
}
```

**Node:** `llm_module` - Executes LLM completions with configurable parameters.

---

#### System Prompt (`system_prompt`)
Injects system instructions and manages tool context.

| Property | Value |
|----------|-------|
| **Type** | Flow Node |
| **Singleton** | No |
| **Order** | 2 |

**Configuration:**
```json
{
    "system_prompt": "You are NeuroCore, a helpful AI assistant.",
    "enabled_tools": []
}
```

**Node:** `system_prompt` - Prepends system message with tool definitions and context injection.

**Features:**
- Tool context injection
- Memory context integration
- Knowledge context integration
- Reasoning context integration
- Plan context integration

---

#### Agent Loop (`agent_loop`)
Autonomous agent with tool execution looping.

| Property | Value |
|----------|-------|
| **Type** | Node Provider |
| **Order** | 1 |

**Configuration:**
```json
{
    "max_iterations": 10,
    "max_tokens": 2048,
    "temperature": 0.7,
    "max_llm_retries": 3,
    "retry_delay": 1.0,
    "tool_error_strategy": "continue",
    "timeout": 120,
    "include_plan_in_context": true,
    "include_memory_context": true,
    "include_knowledge_context": true,
    "include_reasoning_context": true
}
```

**Node:** `agent_loop` - Autonomous agent that loops between LLM and tools until completion.

**Features:**
- Exponential backoff retry
- Tool error handling (continue/stop)
- Timeout protection
- Full execution trace
- Reflection-driven retry support

---

#### Task Planner (`planner`)
Breaks down complex requests into actionable steps.

| Property | Value |
|----------|-------|
| **Type** | Node Provider |
| **Order** | 5 |

**Configuration:**
```json
{
    "max_steps": 10,
    "enabled": true,
    "default_planner_prompt": "You are a task planner..."
}
```

**Node:** `planner` - Creates step-by-step execution plans from user requests.

**Output:**
- `plan` - Array of steps
- `current_step` - Starting step (0)
- `plan_context` - Formatted plan for system prompt
- `plan_needed` - Boolean flag

---

#### Reflection (`reflection`)
Evaluates agent responses for quality and completeness.

| Property | Value |
|----------|-------|
| **Type** | Flow Node |
| **Order** | 4 |

**Configuration:**
```json
{
    "inject_improvement": true,
    "default_reflection_prompt": "You are a reflection agent..."
}
```

**Node:** `reflection` - Evaluates if agent response satisfies user request.

**Output:**
- `satisfied` - Boolean for Conditional Router
- `reflection` - Detailed evaluation object
- `messages` - With improvement feedback (if not satisfied)

---

### 🔄 Control Flow Modules

#### Logic & Flow (`logic`)
Control flow nodes for branching and transformation.

| Property | Value |
|----------|-------|
| **Type** | Node Provider |
| **Order** | 3 |

**Nodes:**

| Node ID | Name | Description | Configurable |
|---------|------|-------------|--------------|
| `trigger_node` | Trigger | Pass-through manual trigger | No |
| `delay_node` | Delay | Pause execution (seconds) | Yes |
| `script_node` | Python Script | Custom Python execution | Yes |
| `repeater_node` | Repeater | Re-trigger flow with delay | Yes |
| `conditional_router` | Conditional Router | Branch based on conditions | Yes |
| `schedule_start_node` | Scheduled Start | Wait until specific time | Yes |

**Conditional Router Options:**
- `check_field`: Field to check (`tool_calls`, `satisfied`, `requires_continuation`)
- `true_branches`: Node IDs for true condition
- `false_branches`: Node IDs for false condition
- `invert`: Invert the condition

---

### 💾 Memory & Knowledge Modules

#### Long-Term Memory (`memory`)
Vector-based memory storage with FAISS and SQLite.

| Property | Value |
|----------|-------|
| **Type** | Node Provider |
| **Order** | 10 |

**Configuration:**
```json
{
    "recall_limit": 3,
    "recall_min_score": 0.3,
    "save_confidence_threshold": 0.75,
    "save_default_confidence": 1.0,
    "save_delay": 3.0,
    "similarity_threshold": 0.9,
    "consolidation_threshold": 0.92,
    "auto_consolidation_hours": 24,
    "belief_ttl_days": 30,
    "recall_access_weight": 0.1,
    "arbiter_model": "",
    "arbiter_prompt": "Extract facts from conversation..."
}
```

**Nodes:**
- `memory_recall` - Searches memory and injects context
- `memory_save` - Stores interactions with smart extraction
- `check_goal` - Retrieves active goals

**Memory Types:** BELIEF, FACT, RULE, EXPERIENCE, PREFERENCE, IDENTITY

---

#### Knowledge Base (`knowledge_base`)
RAG system for document upload and querying.

| Property | Value |
|----------|-------|
| **Type** | Node Provider |
| **Order** | 8 |

**Node:** `query_knowledge` - Retrieves relevant context from uploaded documents.

**Features:**
- Hybrid search (vector + keyword)
- PDF, TXT, MD support
- Chunking and embedding
- Source attribution

---

#### Reasoning Book (`reasoning_book`)
Logs AI's internal thoughts and reasoning steps.

| Property | Value |
|----------|-------|
| **Type** | Node Provider |
| **Order** | 9 |

**Nodes:**
- `reasoning_save` - Saves agent responses as thoughts
- `reasoning_load` - Loads recent thoughts into context

**Configuration:**
```json
{
    "source_field": "content",
    "source": "Flow Node",
    "last_n": 5
}
```

---

#### Memory Browser (`memory_browser`)
UI for viewing and managing long-term memories.

| Property | Value |
|----------|-------|
| **Type** | Service Module |
| **Order** | 11 |

**Features:**
- Search and filter memories
- View memory metadata
- Delete individual memories
- Browse by type and source

---

#### Skills (`skills`)
Manage instruction files (SKILL.md) containing best practices, patterns, and guidelines for AI tasks.

| Property | Value |
|----------|-------|
| **Type** | Service Module |
| **Order** | 7 |

**Features:**
- Import and export skill files
- Inject skills into system prompts
- Skill repository management
- Markdown-based instruction sets

---

### 🛠️ Tool & Integration Modules

#### Tool Library (`tools`)
Manages custom Python tools for AI agents.

| Property | Value |
|----------|-------|
| **Type** | Node Provider |
| **Order** | 6 |

**Node:** `tool_dispatcher` - Executes tool calls from LLM.

**Features:**
- 23 built-in tools
- Custom tool creation
- JSON Schema parameters
- Import/export functionality
- Tool configuration UI

---

#### Chat (`chat`)
Interactive AI assistant interface.

| Property | Value |
|----------|-------|
| **Type** | Node Provider |
| **Order** | 12 |

**Configuration:**
```json
{
    "auto_rename_turns": 3,
    "auto_compact_tokens": 0,
    "compact_keep_last": 10
}
```

**Nodes:**
- `chat_input` - Provides user input and history (singleton)
- `chat_output` - Formats final response (singleton)

---

#### Calendar (`calendar`)
Event scheduling and reminders.

| Property | Value |
|----------|-------|
| **Type** | Node Provider |
| **Order** | 13 |

**Node:** `calendar_watcher` - Checks for events at current time.

**Features:**
- Event creation and management
- Reminder notifications
- Recurring events support

---

#### Telegram Bridge (`telegram`)
Connects NeuroCore to Telegram for remote interaction.

| Property | Value |
|----------|-------|
| **Type** | Node Provider |
| **Order** | 14 |

**Configuration:**
```json
{
    "bot_token": "",
    "chat_id": 0
}
```

**Nodes:**
- `telegram_input` - Receives Telegram messages (singleton)
- `telegram_output` - Sends responses to Telegram (singleton)

---

#### Annotations (`annotations`)
Comment and organization nodes for flows.

| Property | Value |
|----------|-------|
| **Type** | Node Provider |
| **Order** | 7 |

**Node:** `comment_node` - Resizable text box for flow documentation.

---

## 8. Node Implementation Patterns

### Pattern 1: Pass-Through Processor

```python
class ProcessorExecutor:
    async def receive(self, input_data: dict, config: dict = None) -> dict:
        if input_data is None:
            return None
        # Modify data
        input_data['processed'] = True
        return input_data

    async def send(self, processed_data: dict) -> dict:
        return processed_data
```

### Pattern 2: Context Injector

```python
class ContextInjectorExecutor:
    async def receive(self, input_data: dict, config: dict = None) -> dict:
        if input_data is None:
            return None
            
        context = "Additional context here"
        
        # Inject into messages or as separate field
        if 'messages' in input_data:
            input_data['messages'].insert(0, {
                'role': 'system',
                'content': context
            })
        
        return input_data
```

### Pattern 3: Conditional Router

```python
class RouterExecutor:
    async def receive(self, input_data: dict, config: dict = None) -> dict:
        if input_data is None:
            return None
            
        config = config or {}
        condition = config.get('check_field', 'some_field')
        
        # Check condition
        is_true = bool(input_data.get(condition))
        
        if config.get('invert', False):
            is_true = not is_true
            
        # Set routing targets
        targets = config.get('true_branches' if is_true else 'false_branches', [])
        input_data['_route_targets'] = targets
        
        return input_data
```

### Pattern 4: Background Task

```python
import asyncio

class BackgroundExecutor:
    async def _background_task(self, data: dict):
        await asyncio.sleep(5)  # Simulate work
        print(f"Background task completed: {data}")

    async def receive(self, input_data: dict, config: dict = None) -> dict:
        if input_data is None:
            return None
            
        # Fire and forget
        asyncio.create_task(self._background_task(input_data.copy()))
        
        return input_data  # Return immediately
```

### Pattern 5: External API Call

```python
import httpx

class APIExecutor:
    async def receive(self, input_data: dict, config: dict = None) -> dict:
        if input_data is None:
            return None
            
        config = config or {}
        api_url = config.get('api_url', 'https://api.example.com')
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(api_url, timeout=10.0)
                data = response.json()
                
            input_data['api_result'] = data
            return input_data
            
        except Exception as e:
            input_data['error'] = f"API call failed: {str(e)}"
            return input_data
```

---

## 9. Advanced Topics

### Dependency Injection

Access core services through `config` parameter:

```python
async def receive(self, input_data: dict, config: dict = None) -> dict:
    config = config or {}
    
    flow_id = config.get('_flow_id')
    node_id = config.get('_node_id')
    
    # Access settings
    from core.settings import settings
    api_key = settings.get('llm_api_key')
    
    return input_data
```

### Error Handling

Implement robust error handling:

```python
async def receive(self, input_data: dict, config: dict = None) -> dict:
    try:
        # Processing logic
        result = self.process(input_data)
        return result
    except ValueError as e:
        return {'error': f'Invalid input: {str(e)}'}
    except Exception as e:
        return {'error': f'Unexpected error: {str(e)}'}
```

### Testing Nodes

Create unit tests for your nodes:

```python
import pytest
from modules.my_module.node import MyNodeExecutor

@pytest.mark.asyncio
async def test_my_node():
    executor = MyNodeExecutor()
    
    input_data = {'messages': [{'role': 'user', 'content': 'Hello'}]}
    config = {'setting': 'value'}
    
    result = await executor.receive(input_data, config)
    
    assert 'processed' in result
    assert result['processed'] is True
```

### State Management

For nodes requiring state:

```python
class StatefulExecutor:
    def __init__(self):
        self._cache = {}
        self._counter = 0
    
    async def receive(self, input_data: dict, config: dict = None) -> dict:
        self._counter += 1
        self._cache[config.get('_node_id')] = input_data
        return input_data
```

---

## 10. Hot-Swapping & Development

NeuroCore supports hot-loading for rapid development.

### Development Workflow

1. **Create Module Files**
   ```bash
   mkdir modules/my_module
   touch modules/my_module/__init__.py
   touch modules/my_module/module.json
   ```

2. **Edit and Save**
   - Modify code in your editor
   - Save files

3. **Enable in UI**
   - Go to **Settings** → **Modules**
   - Find your module in the list
   - Toggle **Enabled** switch
   - Router loads automatically

4. **Iterate**
   - Make changes
   - Disable → Enable module to reload
   - Or restart NeuroCore for complete refresh

### Module Lifecycle

```
Created → Discovered → Enabled → Loaded → Active
   ↑         ↑           ↑         ↑        │
   └─────────┴───────────┴─────────┴────────┘
              (Hot-swap supported)
```

### Best Practices for Development

| Practice | Benefit |
|----------|---------|
| Use `__init__.py` | Proper Python package structure |
| Handle `None` input | Prevents flow crashes |
| Validate config | Graceful degradation |
| Log errors | Easier debugging |
| Use type hints | Better IDE support |

---

## 11. Troubleshooting

### Common Issues

| Symptom | Cause | Solution |
|---------|-------|----------|
| Module not appearing | Missing `module.json` | Create valid metadata file |
| Router not loading | Missing `router` export | Add `from .router import router` to `__init__.py` |
| Node not executing | Wrong `node_type_id` | Match ID in `module.json` and `get_executor_class` |
| `NoneType` errors | Not handling `None` input | Add `if input_data is None: return None` |
| Config not loading | Invalid JSON syntax | Validate `module.json` with JSON linter |
| Changes not applied | Caching issue | Disable → Enable module or restart |

### Debugging Tips

1. **Check Module Discovery**
   ```python
   # In Python console
   from core.module_manager import ModuleManager
   mm = ModuleManager()
   print(mm.get_all_modules())
   ```

2. **Verify Router Registration**
   ```python
   # Check if router is exposed
   from modules.my_module import router
   print(router.routes)
   ```

3. **Test Node Executor**
   ```python
   import asyncio
   from modules.my_module.node import MyExecutor
   
   executor = MyExecutor()
   result = asyncio.run(executor.receive({'test': 'data'}))
   print(result)
   ```

4. **Review Logs**
   - Check console output for import errors
   - Look for `load_error` in module.json
   - Enable debug mode in settings

### Getting Help

- Review existing modules in `modules/` for examples
- Check test files in `tests/` for usage patterns
- Enable debug logging in NeuroCore settings
- Consult the [NeuroCore Documentation](https://docs.neurocore.ai)

---

> 💡 **Pro Tip:** Start by copying an existing simple module (like `annotations`) as a template, then modify it for your needs!

</diff>
