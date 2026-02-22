import os
import json
from datetime import datetime
from fastapi import APIRouter, Request, Form, UploadFile, File, Query
from fastapi.responses import HTMLResponse, Response, JSONResponse
from fastapi.templating import Jinja2Templates
from core.settings import settings

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
        "full_width_content": True,
        "settings": settings.settings
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
    code: str = Form(...),
    original_name: str = Form("")
):
    try:
        # Validate JSON parameters
        params_json = json.loads(parameters)
        tools = load_tools()
        
        # Determine enabled state and handle renaming
        name = name.strip()
        if original_name:
            original_name = original_name.strip()
            
        is_enabled = True
        
        if original_name and original_name != name:
            if original_name in tools:
                is_enabled = tools[original_name].get("enabled", True)
                del tools[original_name]
                old_code_path = os.path.join(LIBRARY_DIR, f"{original_name}.py")
                if os.path.exists(old_code_path):
                    os.remove(old_code_path)
        elif name in tools:
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
    definitions = []
    for name, tool in tools.items():
        if tool.get("enabled", True):
            definition = tool["definition"]
            # Inject current time into SaveReminder description to help LLM with relative dates
            if name == "SaveReminder":
                now = datetime.now().strftime("%Y-%m-%d %H:%M %A")
                desc = definition["function"].get("description", "")
                definition["function"]["description"] = f"{desc} Current date/time: {now}."
            definitions.append(definition)
    return definitions

@router.get("/export")
async def export_tools(name: str = Query(None)):
    tools = load_tools()
    export_data = []
    
    if name:
        if name not in tools:
            return Response(status_code=404, content="Tool not found")
        items_to_export = [(name, tools[name])]
        filename = f"{name}.json"
    else:
        items_to_export = tools.items()
        filename = "neurocore_tools.json"
    
    for name, data in items_to_export:
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
        headers={"Content-Disposition": f'attachment; filename="{filename}"'}
    )

@router.get("/config/{name}", response_class=HTMLResponse)
async def get_tool_config(request: Request, name: str):
    """Returns the configuration form for a tool."""
    if name == "SendEmail":
        smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
        smtp_port = os.getenv("SMTP_PORT", "587")
        smtp_email = os.getenv("SMTP_EMAIL", "")
        smtp_password = os.getenv("SMTP_PASSWORD", "")
        
        return templates.TemplateResponse(request, "tool_config_sendemail.html", {
            "smtp_server": smtp_server,
            "smtp_port": smtp_port,
            "smtp_email": smtp_email,
            "smtp_password": smtp_password
        })
    
    return Response(status_code=404, content="Configuration not available for this tool")

@router.post("/config/{name}", response_class=HTMLResponse)
async def save_tool_config(request: Request, name: str):
    """Saves the configuration for a tool."""
    try:
        form_data = await request.form()
        
        if name == "SendEmail":
            smtp_server = form_data.get("smtp_server", "smtp.gmail.com")
            smtp_port = form_data.get("smtp_port", "587")
            smtp_email = form_data.get("smtp_email", "")
            smtp_password = form_data.get("smtp_password", "")
            
            os.environ["SMTP_SERVER"] = smtp_server
            os.environ["SMTP_PORT"] = smtp_port
            os.environ["SMTP_EMAIL"] = smtp_email
            os.environ["SMTP_PASSWORD"] = smtp_password
            
            with open(".env.local", "a") as f:
                f.write(f"\nSMTP_SERVER={smtp_server}\n")
                f.write(f"SMTP_PORT={smtp_port}\n")
                f.write(f"SMTP_EMAIL={smtp_email}\n")
                f.write(f"SMTP_PASSWORD={smtp_password}\n")
            
            return Response(content="<div class='text-center py-8 text-emerald-400 font-semibold'>Configuration saved successfully!</div>", status_code=200)
        
        return Response(status_code=404, content="Configuration not available for this tool")
    except Exception as e:
        return Response(status_code=400, content=f"<div class='text-center py-8 text-red-500'>Error saving configuration: {str(e)}</div>")

@router.post("/import")
async def import_tools(request: Request, file: UploadFile = File(...)):
    try:
        content = await file.read()
        filename = file.filename
        tools = load_tools()
        
        if filename.endswith(".json"):
            import_data = json.loads(content)
            
            if isinstance(import_data, dict):
                import_data = [import_data]
            
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

        elif filename.endswith(".py"):
            name = os.path.splitext(os.path.basename(filename))[0]
            code = content.decode("utf-8")
            
            if name not in tools:
                tools[name] = {
                    "definition": {
                        "type": "function",
                        "function": {
                            "name": name,
                            "description": "Imported Python Tool",
                            "parameters": {"type": "object", "properties": {}}
                        }
                    },
                    "enabled": True
                }
            
            with open(os.path.join(LIBRARY_DIR, f"{name}.py"), "w") as f:
                f.write(code)
        else:
            return Response(status_code=400, headers={"HX-Trigger": json.dumps({"showMessage": {"level": "error", "message": "Unsupported file type. Please upload .json or .py"}})})
                
        save_tools(tools)
        
        return await tools_list(request)
    except Exception as e:
        return Response(status_code=400, headers={"HX-Trigger": json.dumps({"showMessage": {"level": "error", "message": f"Import failed: {str(e)}"}})})