import os
import json
from fastapi import APIRouter, Request, Form, UploadFile, File
from fastapi.responses import HTMLResponse, Response, JSONResponse
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
        
        # Preserve enabled state
        is_enabled = True
        if name in tools:
            is_enabled = tools[name].get("enabled", True)
        
        # OpenAI Function Calling Format
        tools[name] = {
            "definition": {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": params_json
                }
            },
            "enabled": is_enabled
        }
        
        # Save Python logic to standalone file
        with open(os.path.join(LIBRARY_DIR, f"{name}.py"), "w") as f:
            f.write(code)

        save_tools(tools)
        return await tools_list(request)
    except Exception as e:
        return Response(status_code=400, headers={"HX-Trigger": json.dumps({"showMessage": {"level": "error", "message": f"Failed to save tool: {str(e)}"}})})

@router.post("/{name}/toggle")
async def toggle_tool(request: Request, name: str):
    tools = load_tools()
    if name in tools:
        current_state = tools[name].get("enabled", True)
        tools[name]["enabled"] = not current_state
        save_tools(tools)
    return await tools_list(request)

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
        if not tool_data.get("enabled", True):
            continue
        desc = ""
        if "definition" in tool_data and "function" in tool_data["definition"]:
            desc = tool_data["definition"]["function"].get("description", "")
        result.append({"name": name, "description": desc})
    return result

@router.get("/definitions")
async def get_definitions():
    """Returns the JSON definitions for all tools to be used in LLM config."""
    tools = load_tools()
    return [t["definition"] for t in tools.values() if t.get("enabled", True)]

@router.get("/export")
async def export_tools():
    tools = load_tools()
    export_data = []
    
    for name, data in tools.items():
        tool_export = {
            "name": name,
            "enabled": data.get("enabled", True),
            "description": data["definition"]["function"]["description"],
            "parameters": data["definition"]["function"]["parameters"]
        }
        
        # Read code
        code_path = os.path.join(LIBRARY_DIR, f"{name}.py")
        if os.path.exists(code_path):
            with open(code_path, "r") as f:
                tool_export["code"] = f.read()
        else:
            tool_export["code"] = ""
            
        export_data.append(tool_export)
        
    return JSONResponse(
        content=export_data,
        headers={"Content-Disposition": 'attachment; filename="neurocore_tools.json"'}
    )

@router.post("/import")
async def import_tools(request: Request, file: UploadFile = File(...)):
    try:
        content = await file.read()
        import_data = json.loads(content)
        
        if isinstance(import_data, dict):
            import_data = [import_data]
            
        tools = load_tools()
        
        for tool_data in import_data:
            name = tool_data.get("name")
            if not name: continue
            
            # Update tools.json structure
            tools[name] = {
                "definition": {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": tool_data.get("description", ""),
                        "parameters": tool_data.get("parameters", {"type": "object", "properties": {}})
                    }
                },
                "enabled": tool_data.get("enabled", True)
            }
            
            # Save code
            code = tool_data.get("code", "")
            with open(os.path.join(LIBRARY_DIR, f"{name}.py"), "w") as f:
                f.write(code)
                
        save_tools(tools)
        
        return await tools_list(request)
    except Exception as e:
        return Response(status_code=400, headers={"HX-Trigger": json.dumps({"showMessage": {"level": "error", "message": f"Import failed: {str(e)}"}})})