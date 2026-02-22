import os
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from core.module_manager import ModuleManager
from core.routers import router as core_router
from core.settings import settings
from core.flow_manager import flow_manager
from core.flow_runner import FlowRunner

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles startup and shutdown events for the application."""
    # Initialize the module manager and load enabled modules
    module_manager = ModuleManager(app=app)
    app.state.module_manager = module_manager
    module_manager.load_enabled_modules()

    # Auto-start active flow if it contains a Repeater node
    active_flow_id = settings.get("active_ai_flow")
    if active_flow_id:
        flow = flow_manager.get_flow(active_flow_id)
        if flow:
            # Define node types that should auto-start in the background
            # This allows multiple independent chains (e.g. Repeater, Cron, Event Watcher) to run simultaneously
            background_node_types = ["repeater_node"]
            
            # Find all nodes of these types
            start_nodes = [n for n in flow.get("nodes", []) if n.get("nodeTypeId") in background_node_types]
            
            for node in start_nodes:
                print(f"[System] Auto-starting flow '{flow.get('name')}' from {node['nodeTypeId']} '{node['id']}'.")
                # Start with _repeat_count=1 to skip Chat Input nodes and prevent ghost replies
                runner = FlowRunner(active_flow_id)
                asyncio.create_task(runner.run({"_repeat_count": 1}, start_node_id=node['id']))

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
    # Exclude all .json files from the reloader. This prevents the server from
    # restarting when state files (settings, flows, sessions, module configs)
    # are updated by the application itself, which fixes the module reordering issue.
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, reload_excludes=["*.json", "modules/*/*.json"])
