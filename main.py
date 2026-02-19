from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Form, Depends, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, RedirectResponse
from core.settings import SettingsManager
from core.dependencies import get_settings_manager, get_llm_bridge
from core.module_manager import ModuleManager
from core.llm import LLMBridge

# Initialize core components
module_manager = ModuleManager()
loaded_modules_meta = []

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles startup and shutdown events for the application."""
    global loaded_modules_meta
    loaded_modules_meta = module_manager.load_enabled_modules(app)
    yield
    # Add shutdown logic here if needed in the future

app = FastAPI(title="NeuroCore", description="Modular LLM API Core", lifespan=lifespan)

templates = Jinja2Templates(directory="web/templates")
app.mount("/static", StaticFiles(directory="web/static"), name="static")

@app.get("/navbar", response_class=HTMLResponse)
async def get_navbar(request: Request):
    return templates.TemplateResponse(request, "navbar.html", {"modules": module_manager.get_all_modules()})

@app.get("/modules/list", response_class=HTMLResponse)
async def get_modules_list(request: Request):
    return templates.TemplateResponse(request, "module_list.html", {"modules": module_manager.get_all_modules()})

@app.get("/llm-status", response_class=HTMLResponse)
async def get_llm_status(request: Request, llm: LLMBridge = Depends(get_llm_bridge)):
    api_status_check = await llm.get_models()
    api_online = "error" not in api_status_check
    return f"""
    <div class="flex items-center space-x-2">
        <div class="w-2 h-2 rounded-full {"bg-emerald-500" if api_online else "bg-red-500"}"></div>
        <span class="text-xs {"text-slate-400" if api_online else "text-red-400"}">LLM Status: {"Online" if api_online else "Offline"}</span>
    </div>
    """

# --- Core Routes ---
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    enabled_modules = [m for m in module_manager.get_all_modules() if m.get("enabled")]

    return templates.TemplateResponse(request, "index.html", {
        "modules": enabled_modules,
        "active_module": None
    })

@app.get("/chat", response_class=HTMLResponse)
async def chat_page(request: Request):
    enabled_modules = [m for m in module_manager.get_all_modules() if m.get("enabled")]

    return templates.TemplateResponse(request, "index.html", {
        "modules": enabled_modules,
        "active_module": "chat"
    })

@app.get("/settings", response_class=HTMLResponse)
async def get_settings(request: Request, settings: SettingsManager = Depends(get_settings_manager)):
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
@app.get("/modules", response_class=HTMLResponse)
async def get_modules_page(request: Request):
    all_modules = module_manager.get_all_modules()
    return templates.TemplateResponse(request, "modules.html", {"modules": all_modules})

@app.post("/modules/{module_id}/enable")
async def enable_module_route(request: Request, module_id: str):
    module = module_manager.enable_module(module_id)
    if not module:
        raise HTTPException(status_code=404, detail="Module not found")
    
    # Return updated list and trigger navbar refresh
    all_modules = module_manager.get_all_modules()
    return templates.TemplateResponse(
        request,
        "module_list.html",
        {"modules": all_modules},
        headers={"HX-Trigger": "modulesChanged"}
    )

@app.post("/modules/{module_id}/disable")
async def disable_module_route(request: Request, module_id: str):
    module = module_manager.disable_module(module_id)
    if not module:
        raise HTTPException(status_code=404, detail="Module not found")
    
    all_modules = module_manager.get_all_modules()
    return templates.TemplateResponse(
        request,
        "module_list.html",
        {"modules": all_modules},
        headers={"HX-Trigger": "modulesChanged"}
    )

# --- Server Startup ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
