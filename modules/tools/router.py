import os
import json
from fastapi import APIRouter, Request, Form
from fastapi.responses import HTMLResponse, Response
from fastapi.templating import Jinja2Templates

router = APIRouter()
templates = Jinja2Templates(directory="web/templates")
TOOLS_FILE = os.path.join(os.path.dirname(__file__), "tools.json")
LIBRARY_DIR = os.path.join(os.path.dirname(__file__), "library")

os.makedirs(LIBRARY_DIR, exist_ok=True)

def load_tools():
    if os.path.exists(TOOLS_FILE):
        with open(TOOLS_FILE, "r") as f:
            try: return json.load(f)
            except: return {}
    return {}

def save_tools(tools):
    with open(TOOLS_FILE, "w") as f:
        json.dump(tools, f, indent=4)

@router.get("", response_class=HTMLResponse)
async def tools_page(request: Request):
    """Loads the main dashboard with the Tool Library active."""
    module_manager = request.app.state.module_manager
    enabled_modules = [m for m in module_manager.get_all_modules() if m.get("enabled")]
    return templates.TemplateResponse(request, "index.html", {
        "modules": enabled_modules,
        "active_module": "tools",
        "full_width_content": True
    })

@router.get("/gui", response_class=HTMLResponse)
async def tools_gui(request: Request):
    """Returns the Tool Library GUI fragment."""
    tools = load_tools()
    return templates.TemplateResponse(request, "tools_gui.html", {"tools": tools})

@router.get("/list", response_class=HTMLResponse)
async def tools_list(request: Request):
    tools = load_tools()
    return templates.TemplateResponse(request, "tools_list.html", {"tools": tools})

@router.post("/save")
async def save_tool(
    request: Request,
    name: str = Form(...),
    description: str = Form(...),
    parameters: str = Form(...),
    code: str = Form(...)
):
    try:
        # Validate JSON parameters
        params_json = json.loads(parameters)
        tools = load_tools()
        
        # OpenAI Function Calling Format
        tools[name] = {
            "definition": {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": params_json
                }
            }
        }
        
        # Save Python logic to standalone file
        with open(os.path.join(LIBRARY_DIR, f"{name}.py"), "w") as f:
            f.write(code)

        save_tools(tools)
        return await tools_list(request)
    except Exception as e:
        return Response(status_code=400, headers={"HX-Trigger": json.dumps({"showMessage": {"level": "error", "message": f"Failed to save tool: {str(e)}"}})})

@router.delete("/{name}")
async def delete_tool(request: Request, name: str):
    tools = load_tools()
    if name in tools:
        del tools[name]
        # Remove associated python file
        code_path = os.path.join(LIBRARY_DIR, f"{name}.py")
        if os.path.exists(code_path):
            os.remove(code_path)
        save_tools(tools)
    return await tools_list(request)

@router.get("/edit/{name}")
async def edit_tool(name: str):
    """Returns tool metadata and code for editing."""
    tools = load_tools()
    if name not in tools:
        return Response(status_code=404)
    
    tool_data = tools[name]
    code = ""
    code_path = os.path.join(LIBRARY_DIR, f"{name}.py")
    if os.path.exists(code_path):
        with open(code_path, "r") as f:
            code = f.read()
            
    return {
        "name": name,
        "description": tool_data["definition"]["function"]["description"],
        "parameters": json.dumps(tool_data["definition"]["function"]["parameters"], indent=2),
        "code": code
    }

@router.get("/names")
async def get_tool_names():
    """Returns a list of all tool names."""
    tools = load_tools()
    return list(tools.keys())

@router.get("/with-descriptions")
async def get_tools_with_descriptions():
    """Returns all tools with their names and descriptions."""
    tools = load_tools()
    result = []
    for name, tool_data in tools.items():
        desc = ""
        if "definition" in tool_data and "function" in tool_data["definition"]:
            desc = tool_data["definition"]["function"].get("description", "")
        result.append({"name": name, "description": desc})
    return result

@router.get("/definitions")
async def get_definitions():
    """Returns the JSON definitions for all tools to be used in LLM config."""
    tools = load_tools()
    return [t["definition"] for t in tools.values()]