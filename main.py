import os
import asyncio
import sys
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, HTTPException, Header
from fastapi.staticfiles import StaticFiles
from core.module_manager import ModuleManager
from core.routers import router as core_router
from core.settings import settings
from core.flow_manager import flow_manager
from core.flow_runner import FlowRunner
from core.debug import debug_logger
from core.llm import close_shared_client


async def verify_debug_key(x_debug_key: str = Header(...)):
    """Verify debug API key for protected debug endpoints."""
    debug_key = settings.get('debug_api_key')
    if not debug_key or x_debug_key != debug_key:
        raise HTTPException(status_code=403, detail="Invalid or missing debug API key")
    return x_debug_key


def _fire_repeater_nodes(app, flow_id: str, flow: dict, is_auto_start: bool = False):
    """
    Fire all repeater nodes in a flow.
    
    Args:
        app: FastAPI app instance
        flow_id: The flow ID
        flow: The flow data dictionary
        is_auto_start: If True, this is an auto-start from lifespan (don't await)
    
    Returns:
        List of created tasks
    """
    background_node_types = ["repeater_node"]
    nodes = flow.get("nodes", [])
    connections = flow.get("connections", [])
    created_tasks = []
    
    for node in nodes:
        if node.get("nodeTypeId") in background_node_types:
            has_incoming = any(c.get("to") == node.get("id") for c in connections)
            if has_incoming:
                event_type = "auto_start" if is_auto_start else "manual_trigger"
                print(f"[{'System' if is_auto_start else 'Debug'}] {'Auto-starting' if is_auto_start else 'Manually firing'} flow '{flow.get('name')}' from {node['nodeTypeId']} '{node['id']}'.")
                if settings.get("debug_mode"):
                    debug_logger.log(flow_id, node['id'], node.get('name'), event_type, {})
                runner = FlowRunner(flow_id)
                task = asyncio.create_task(runner.run({"_repeat_count": 1}, start_node_id=node['id']))
                
                if hasattr(app.state, "background_tasks"):
                    app.state.background_tasks.add(task)
                    task.add_done_callback(lambda t, fid=flow_id, nid=node['id']: background_task_callback(t, fid, nid))
                created_tasks.append(task)
    
    return created_tasks

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

    # Auto-start all active flows that contain Repeater nodes
    active_flow_ids = settings.get("active_ai_flows", [])
    for active_flow_id in active_flow_ids:
        flow = flow_manager.get_flow(active_flow_id)
        if flow:
            _fire_repeater_nodes(app, active_flow_id, flow, is_auto_start=True)

    yield
    
    # Graceful shutdown: Cancel all background tasks
    print("[System] Shutting down, cancelling background tasks...")
    if hasattr(app.state, "background_tasks"):
        # Iterate over a copy to avoid mutation during iteration
        for task in list(app.state.background_tasks):
            if not task.done():
                task.cancel()
                print(f"[System] Cancelled task for flow")
        
        # Wait for tasks to complete cancellation (with timeout)
        if app.state.background_tasks:
            done, pending = await asyncio.wait(
                app.state.background_tasks, 
                timeout=5.0,
                return_when=asyncio.ALL_COMPLETED
            )
            if pending:
                print(f"[System] {len(pending)} tasks did not complete in time")
    
    # Shutdown module managers if they have cleanup methods
    if hasattr(app.state, "module_manager"):
        module_manager = app.state.module_manager
        # Call shutdown on any modules that support it
        for module in module_manager.get_all_modules():
            if module.get("enabled"):
                # Try to gracefully stop the module
                mod = sys.modules.get(f"modules.{module['id']}")
                if mod and hasattr(mod, 'shutdown'):
                    await mod.shutdown()
                print(f"[System] Stopping module: {module.get('name', 'unknown')}")
    
    # Close the shared LLM client
    await close_shared_client()

app = FastAPI(title="NeuroCore", description="Modular LLM API Core", lifespan=lifespan)

# Ensure static directory exists to prevent startup errors
os.makedirs("web/static", exist_ok=True)
app.mount("/static", StaticFiles(directory="web/static"), name="static")
app.include_router(core_router)

# Debug Endpoint to manually fire all active flows
@app.post("/debug/fire-flow", dependencies=[Depends(verify_debug_key)])
async def debug_fire_flow():
    active_flow_ids = settings.get("active_ai_flows", [])
    for active_flow_id in active_flow_ids:
        flow = flow_manager.get_flow(active_flow_id)
        if flow:
            tasks = _fire_repeater_nodes(app, active_flow_id, flow, is_auto_start=False)
            if tasks:
                return {"status": "fired", "node": tasks[0]}
    return {"status": "failed"}

# --- Server Startup ---
if __name__ == "__main__":
    import uvicorn
    # Exclude all .json files from the reloader. This prevents the server from
    # restarting when state files (settings, flows, sessions, module configs)
    # are updated by the application itself, which fixes the module reordering issue.
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, reload_excludes=["*.json", "data/*.json", "modules/*/*.json"])
