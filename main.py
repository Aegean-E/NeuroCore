from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Form, Depends, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, RedirectResponse
from core.settings import SettingsManager
from core.dependencies import get_settings_manager
from core.module_manager import ModuleManager

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

# --- Core Routes ---
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse(request, "index.html", {"modules": loaded_modules_meta})

@app.get("/settings", response_class=HTMLResponse)
async def get_settings(request: Request, settings: SettingsManager = Depends(get_settings_manager)):
    return templates.TemplateResponse(request, "settings.html", {"settings": settings.settings})

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
async def enable_module_route(module_id: str):
    module = module_manager.enable_module(module_id)
    if not module:
        raise HTTPException(status_code=404, detail="Module not found")
    return RedirectResponse(url="/modules", status_code=303)

@app.post("/modules/{module_id}/disable")
async def disable_module_route(module_id: str):
    module = module_manager.disable_module(module_id)
    if not module:
        raise HTTPException(status_code=404, detail="Module not found")
    return RedirectResponse(url="/modules", status_code=303)

# --- Server Startup ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
