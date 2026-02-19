from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Form, Depends, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from core.settings import SettingsManager
from core.dependencies import get_settings_manager, get_llm_bridge
from core.module_manager import ModuleManager
from core.llm import LLMBridge
from core.settings import settings
from core.flow_manager import flow_manager

# Dependency to get the module manager instance from the app state
def get_module_manager(request: Request) -> ModuleManager:
    return request.app.state.module_manager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles startup and shutdown events for the application."""
    # Initialize the module manager and load enabled modules
    module_manager = ModuleManager(app=app)
    app.state.module_manager = module_manager
    module_manager.load_enabled_modules()
    yield
    # Add shutdown logic here if needed in the future

app = FastAPI(title="NeuroCore", description="Modular LLM API Core", lifespan=lifespan)

templates = Jinja2Templates(directory="web/templates")
app.mount("/static", StaticFiles(directory="web/static"), name="static")

@app.get("/navbar", response_class=HTMLResponse)
async def get_navbar(request: Request, module_manager: ModuleManager = Depends(get_module_manager)):
    return templates.TemplateResponse(request, "navbar.html", {"modules": module_manager.get_all_modules()})

@app.get("/llm-status", response_class=HTMLResponse)
async def get_llm_status(request: Request, llm: LLMBridge = Depends(get_llm_bridge)):
    api_status_check = await llm.get_models()
    api_online = "error" not in api_status_check
    return f"""
    <div class="flex items-center space-x-2">
        <div class="w-2 h-2 rounded-full {"bg-emerald-500 animate-pulse" if api_online else "bg-red-500"}"></div>
        <span class="text-xs {"text-slate-400" if api_online else "text-red-400"}">LLM Status: {"Online" if api_online else "Offline"}</span>
    </div>
    """

# --- Core Routes ---
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request, module_manager: ModuleManager = Depends(get_module_manager)):
    # The root is the module browser.
    all_modules = module_manager.get_all_modules()

    return templates.TemplateResponse(request, "index.html", {
        "modules": all_modules,
        "active_module": None
    })

@app.get("/modules/list", response_class=HTMLResponse)
async def list_modules(request: Request, module_manager: ModuleManager = Depends(get_module_manager)):
    all_modules = module_manager.get_all_modules()
    return templates.TemplateResponse(request, "module_list.html", {"modules": all_modules})

@app.get("/chat", response_class=HTMLResponse)
async def chat_page(request: Request, module_manager: ModuleManager = Depends(get_module_manager)):
    from modules.chat.sessions import session_manager
    enabled_modules = [m for m in module_manager.get_all_modules() if m.get("enabled")]
    sessions = session_manager.list_sessions()

    return templates.TemplateResponse(request, "index.html", {
        "modules": enabled_modules,
        "active_module": "chat",
        "sessions": sessions
    })

@app.get("/ai-flow", response_class=HTMLResponse)
async def ai_flow_page(request: Request, module_manager: ModuleManager = Depends(get_module_manager)):
    all_modules = module_manager.get_all_modules()
    active_flow_id = settings.get("active_ai_flow")
    flows = flow_manager.list_flows()
    return templates.TemplateResponse(request, "ai_flow.html", {"modules": all_modules, "flows": flows, "active_flow_id": active_flow_id})

@app.get("/ai-flow/{flow_id}", response_class=JSONResponse)
async def get_flow_data(flow_id: str):
    flow = flow_manager.get_flow(flow_id)
    if not flow:
        raise HTTPException(status_code=404, detail="Flow not found")
    return flow

@app.post("/ai-flow/save")
async def save_ai_flow(
    request: Request,
    name: str = Form(...),
    nodes: str = Form(...),
    connections: str = Form(...)
):
    import json
    nodes_data = json.loads(nodes)
    connections_data = json.loads(connections)
    flow_manager.save_flow(name=name, nodes=nodes_data, connections=connections_data)
    flows = flow_manager.list_flows()
    active_flow_id = settings.get("active_ai_flow")
    return templates.TemplateResponse(request, "ai_flow_list.html", {"flows": flows, "active_flow_id": active_flow_id})

@app.post("/ai-flow/{flow_id}/set-active", response_class=HTMLResponse)
async def set_active_flow(request: Request, flow_id: str, settings_man: SettingsManager = Depends(get_settings_manager)):
    settings_man.save_settings({"active_ai_flow": flow_id})
    flows = flow_manager.list_flows()
    active_flow_id = settings_man.get("active_ai_flow")
    return templates.TemplateResponse(request, "ai_flow_list.html", {"flows": flows, "active_flow_id": active_flow_id})

@app.post("/ai-flow/{flow_id}/delete", response_class=HTMLResponse)
async def delete_flow(request: Request, flow_id: str, settings_man: SettingsManager = Depends(get_settings_manager)):
    # If the deleted flow was the active one, unset it
    if settings_man.get("active_ai_flow") == flow_id:
        settings_man.save_settings({"active_ai_flow": None})

    flow_manager.delete_flow(flow_id)
    
    flows = flow_manager.list_flows()
    active_flow_id = settings_man.get("active_ai_flow")
    return templates.TemplateResponse(request, "ai_flow_list.html", {"flows": flows, "active_flow_id": active_flow_id})

@app.get("/modules/{module_id}/details", response_class=HTMLResponse)
async def get_module_details(request: Request, module_id: str, module_manager: ModuleManager = Depends(get_module_manager)):
    module = module_manager.modules.get(module_id)
    if not module:
        raise HTTPException(status_code=404, detail="Module not found")
    return templates.TemplateResponse(request, "module_details.html", {"module": module})


@app.get("/settings", response_class=HTMLResponse)
async def get_settings(request: Request, settings: SettingsManager = Depends(get_settings_manager), module_manager: ModuleManager = Depends(get_module_manager)):
    return templates.TemplateResponse(request, "settings.html", {
        "settings": settings.settings,
        "modules": module_manager.get_all_modules()
    })

@app.post("/settings/save")
async def save_settings_route(
    llm_api_url: str = Form(...),
    default_model: str = Form(...),
    temperature: float = Form(...),
    max_tokens: int = Form(...),
    settings: SettingsManager = Depends(get_settings_manager)
):
    new_settings = {
        "llm_api_url": llm_api_url,
        "default_model": default_model,
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    settings.save_settings(new_settings)
    return RedirectResponse(url="/settings", status_code=303)

# --- Module Management Routes ---
@app.post("/modules/{module_id}/enable")
async def enable_module_route(request: Request, module_id: str, module_manager: ModuleManager = Depends(get_module_manager)):
    module = module_manager.enable_module(module_id)
    if not module:
        raise HTTPException(status_code=404, detail="Module not found")
    
    # Return updated details view and trigger navbar refresh
    return templates.TemplateResponse(
        request,
        "module_details.html",
        {"module": module},
        headers={"HX-Trigger": "modulesChanged"}
    )

@app.post("/modules/{module_id}/disable")
async def disable_module_route(request: Request, module_id: str, module_manager: ModuleManager = Depends(get_module_manager)):
    module = module_manager.disable_module(module_id)
    if not module:
        raise HTTPException(status_code=404, detail="Module not found")
    
    # Return updated details view and trigger navbar refresh
    return templates.TemplateResponse(
        request,
        "module_details.html",
        {"module": module},
        headers={"HX-Trigger": "modulesChanged"}
    )

# --- Server Startup ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
