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
from core.debug import debug_logger

def background_task_callback(task, flow_id, node_id):
    try:
        app.state.background_tasks.discard(task)
        task.result() # This will raise exception if task failed
        if settings.get("debug_mode"):
             debug_logger.log(flow_id, node_id, "System", "task_finished", {})
    except asyncio.CancelledError:
        print(f"[System] Task for flow {flow_id} cancelled")
    except Exception as e:
        print(f"[System] Task for flow {flow_id} failed: {e}")
        if settings.get("debug_mode"):
             debug_logger.log(flow_id, node_id, "System", "task_failed", {"error": str(e)})

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles startup and shutdown events for the application."""
    # Initialize the module manager and load enabled modules
    module_manager = ModuleManager(app=app)
    app.state.module_manager = module_manager
    module_manager.load_enabled_modules()

    # Initialize background tasks set to prevent GC of fire-and-forget tasks
    app.state.background_tasks = set()

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
                if settings.get("debug_mode"):
                    debug_logger.log(active_flow_id, node['id'], node.get('name'), "auto_start", {})
                # Start with _repeat_count=1 to skip Chat Input nodes and prevent ghost replies
                runner = FlowRunner(active_flow_id)
                task = asyncio.create_task(runner.run({"_repeat_count": 1}, start_node_id=node['id']))
                app.state.background_tasks.add(task)
                task.add_done_callback(lambda t, fid=active_flow_id, nid=node['id']: background_task_callback(t, fid, nid))

    yield
    # Add shutdown logic here if needed in the future

app = FastAPI(title="NeuroCore", description="Modular LLM API Core", lifespan=lifespan)

# Ensure static directory exists to prevent startup errors
os.makedirs("web/static", exist_ok=True)
app.mount("/static", StaticFiles(directory="web/static"), name="static")
app.include_router(core_router)

# Debug Endpoint to manually fire the active flow
@app.post("/debug/fire-flow")
async def debug_fire_flow():
    active_flow_id = settings.get("active_ai_flow")
    if active_flow_id:
        flow = flow_manager.get_flow(active_flow_id)
        if flow:
            background_node_types = ["repeater_node"]
            start_nodes = [n for n in flow.get("nodes", []) if n.get("nodeTypeId") in background_node_types]
            
            if start_nodes:
                node = start_nodes[0]
                print(f"[Debug] Manually firing flow '{flow.get('name')}' from {node['nodeTypeId']} '{node['id']}'.")
                if settings.get("debug_mode"):
                    debug_logger.log(active_flow_id, node['id'], node.get('name'), "manual_trigger", {})
                runner = FlowRunner(active_flow_id)
                task = asyncio.create_task(runner.run({"_repeat_count": 1}, start_node_id=node['id']))
                
                if hasattr(app.state, "background_tasks"):
                    app.state.background_tasks.add(task)
                    task.add_done_callback(lambda t, fid=active_flow_id, nid=node['id']: background_task_callback(t, fid, nid))
                
                return {"status": "fired", "node": node['id']}
    return {"status": "failed"}

# --- Server Startup ---
if __name__ == "__main__":
    import uvicorn
    # Exclude all .json files from the reloader. This prevents the server from
    # restarting when state files (settings, flows, sessions, module configs)
    # are updated by the application itself, which fixes the module reordering issue.
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, reload_excludes=["*.json", "data/*.json", "modules/*/*.json"])
