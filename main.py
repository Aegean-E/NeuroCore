import os
import importlib
import pkgutil
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles startup and shutdown events for the application."""
    load_modules()
    yield
    # Add shutdown logic here if needed in the future

app = FastAPI(title="NeuroCore", description="Modular LLM API Core", lifespan=lifespan)

# Setup templates and static files
templates = Jinja2Templates(directory="web/templates")
app.mount("/static", StaticFiles(directory="web/static"), name="static")

# Module management
loaded_modules = []

def load_modules():
    """Dynamically loads modules from the 'modules' directory."""
    modules_path = os.path.join(os.path.dirname(__file__), "modules")
    if not os.path.exists(modules_path):
        os.makedirs(modules_path)
    
    for _, name, ispkg in pkgutil.iter_modules([modules_path]):
        if ispkg:
            try:
                module = importlib.import_module(f"modules.{name}")
                if hasattr(module, "router"):
                    app.include_router(module.router, prefix=f"/{name}", tags=[name])
                    loaded_modules.append(name)
                    print(f"Loaded module: {name}")
            except Exception as e:
                print(f"Failed to load module {name}: {e}")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse(request, "index.html", {"modules": loaded_modules})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
