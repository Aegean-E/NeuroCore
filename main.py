import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from core.module_manager import ModuleManager
from core.routers import router as core_router

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

# Ensure static directory exists to prevent startup errors
os.makedirs("web/static", exist_ok=True)
app.mount("/static", StaticFiles(directory="web/static"), name="static")
app.include_router(core_router)

# --- Server Startup ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
