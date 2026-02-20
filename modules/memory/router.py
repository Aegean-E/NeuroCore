from fastapi import APIRouter, Request, Form
from fastapi.responses import HTMLResponse, Response
from fastapi.templating import Jinja2Templates
import json
import time
import asyncio
from .backend import memory_store
from .consolidation import MemoryConsolidator

router = APIRouter()
templates = Jinja2Templates(directory="web/templates")

@router.get("/stats", response_class=HTMLResponse)
async def get_stats(request: Request):
    try:
        loop = asyncio.get_running_loop()
        stats = await loop.run_in_executor(memory_store.executor, memory_store.get_memory_stats)
    except Exception as e:
        return f"<div class='text-red-400 text-sm p-4 border border-red-500/50 rounded-lg bg-red-900/20'>Error loading stats: {str(e)}</div>"
    
    # Build type breakdown HTML
    type_html = ""
    types_dict = stats.get('types', {})
    for t_name, count in types_dict.items():
        type_html += f'<span class="px-2 py-0.5 bg-slate-700 text-slate-300 rounded text-[10px] font-mono">{t_name}: {count}</span> '

    html = f"""
    <div class="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-4 mb-6">
        <div class="bg-slate-800 p-4 rounded-xl border border-slate-700/50">
            <div class="text-xs text-slate-400 uppercase tracking-wider mb-1">Total Memories</div>
            <div class="text-2xl font-bold text-slate-100">{stats.get('grand_total', 0)}</div>
        </div>
        <div class="bg-slate-800 p-4 rounded-xl border border-slate-700/50">
            <div class="text-xs text-slate-400 uppercase tracking-wider mb-1">Active</div>
            <div class="text-2xl font-bold text-emerald-400">{stats.get('total', 0)}</div>
        </div>
        <div class="bg-slate-800 p-4 rounded-xl border border-slate-700/50">
            <div class="text-xs text-slate-400 uppercase tracking-wider mb-1">User Origin</div>
            <div class="text-2xl font-bold text-blue-400">{stats.get('user', 0)}</div>
        </div>
        <div class="bg-slate-800 p-4 rounded-xl border border-slate-700/50">
            <div class="text-xs text-slate-400 uppercase tracking-wider mb-1">Assistant Origin</div>
            <div class="text-2xl font-bold text-indigo-400">{stats.get('assistant', 0)}</div>
        </div>
        <div class="bg-slate-800 p-4 rounded-xl border border-slate-700/50">
            <div class="text-xs text-slate-400 uppercase tracking-wider mb-1">Consolidated</div>
            <div class="text-2xl font-bold text-purple-400">{stats.get('consolidated', 0)}</div>
        </div>
    </div>
    <div class="flex flex-wrap gap-2 items-center">
        <span class="text-xs text-slate-500 uppercase font-bold tracking-tighter">Type Breakdown:</span>
        {type_html if type_html else '<span class="text-xs text-slate-600 italic">None</span>'}
    </div>
    """
    
    if stats.get('grand_total', 0) == 0:
        html = "<p class='text-slate-500 italic'>No memories stored yet.</p>"
        
    return html

@router.post("/settings/save")
async def save_memory_settings(
    request: Request, 
    recall_limit: int = Form(None), 
    recall_min_score: float = Form(None),
    save_confidence_threshold: float = Form(None),
    consolidation_threshold: float = Form(None),
    auto_consolidation_hours: float = Form(None),
    arbiter_model: str = Form(None),
    arbiter_prompt: str = Form(None)
):
    module_manager = request.app.state.module_manager
    memory_module = module_manager.modules.get("memory")
    if not memory_module:
        return Response(status_code=404)
        
    config = memory_module.get("config", {}).copy()
    if recall_limit is not None: config["recall_limit"] = recall_limit
    if recall_min_score is not None: config["recall_min_score"] = recall_min_score
    if save_confidence_threshold is not None: config["save_confidence_threshold"] = save_confidence_threshold
    if consolidation_threshold is not None: config["consolidation_threshold"] = consolidation_threshold
    if auto_consolidation_hours is not None: config["auto_consolidation_hours"] = auto_consolidation_hours
    if arbiter_model is not None: config["arbiter_model"] = arbiter_model
    if arbiter_prompt is not None: config["arbiter_prompt"] = arbiter_prompt
    
    module_manager.update_module_config("memory", config)
    
    return Response(status_code=200, headers={"HX-Trigger": json.dumps({"showMessage": {"level": "success", "message": "Memory settings saved"}})})

@router.post("/consolidate")
async def consolidate_memories(request: Request):
    module_manager = request.app.state.module_manager
    memory_module = module_manager.modules.get("memory")
    config = memory_module.get("config", {}) if memory_module else {}
    
    consolidator = MemoryConsolidator(config=config)
    count = await consolidator.run()
    memory_store.last_consolidation_ts = time.time()
    
    return Response(status_code=200, headers={"HX-Trigger": json.dumps({"showMessage": {"level": "success", "message": f"Consolidated {count} redundant memories"}})})

@router.post("/wipe")
async def wipe_memories(request: Request):
    try:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(memory_store.executor, memory_store.wipe_all)
        return Response(status_code=200, headers={"HX-Trigger": json.dumps({"memoryWiped": None, "showMessage": {"level": "success", "message": "All memories wiped successfully"}})})
    except Exception as e:
        return Response(status_code=500, headers={"HX-Trigger": json.dumps({"showMessage": {"level": "error", "message": f"Failed to wipe memories: {str(e)}"}})})