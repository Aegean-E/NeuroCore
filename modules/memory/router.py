from fastapi import APIRouter, Request, Form
from fastapi.responses import HTMLResponse, Response
from fastapi.templating import Jinja2Templates
import json
from .backend import memory_store

router = APIRouter()
templates = Jinja2Templates(directory="web/templates")

@router.get("/stats", response_class=HTMLResponse)
async def get_stats(request: Request):
    stats = memory_store.get_memory_stats()
    
    html = """
    <div class="grid grid-cols-2 gap-4 mb-6">
        <div class="bg-slate-800 p-4 rounded-xl border border-slate-700/50">
            <div class="text-xs text-slate-400 uppercase tracking-wider mb-1">Total Active</div>
            <div class="text-2xl font-bold text-slate-100">{}</div>
        </div>
        <div class="bg-slate-800 p-4 rounded-xl border border-slate-700/50">
            <div class="text-xs text-slate-400 uppercase tracking-wider mb-1">Archived</div>
            <div class="text-2xl font-bold text-slate-100">{}</div>
        </div>
        <div class="bg-slate-800 p-4 rounded-xl border border-slate-700/50">
            <div class="text-xs text-slate-400 uppercase tracking-wider mb-1">User Origin</div>
            <div class="text-2xl font-bold text-blue-400">{}</div>
        </div>
        <div class="bg-slate-800 p-4 rounded-xl border border-slate-700/50">
            <div class="text-xs text-slate-400 uppercase tracking-wider mb-1">Assistant Origin</div>
            <div class="text-2xl font-bold text-emerald-400">{}</div>
        </div>
    </div>
    """.format(stats.get('total', 0), stats.get('archived', 0), stats.get('user', 0), stats.get('assistant', 0))
    
    if stats.get('total', 0) == 0:
        html = "<p class='text-slate-500 italic'>No memories stored yet.</p>"
        
    return html

@router.post("/settings/recall")
async def save_recall_settings(request: Request, recall_limit: int = Form(...), recall_min_score: float = Form(...)):
    module_manager = request.app.state.module_manager
    memory_module = module_manager.modules.get("memory")
    if not memory_module:
        return Response(status_code=404)
        
    config = memory_module.get("config", {}).copy()
    config["recall_limit"] = recall_limit
    config["recall_min_score"] = recall_min_score
    
    module_manager.update_module_config("memory", config)
    
    return Response(status_code=200, headers={"HX-Trigger": json.dumps({"showMessage": {"level": "success", "message": "Recall settings saved"}})})

@router.post("/settings/persistence")
async def save_persistence_settings(request: Request, save_confidence_threshold: float = Form(...)):
    module_manager = request.app.state.module_manager
    memory_module = module_manager.modules.get("memory")
    if not memory_module:
        return Response(status_code=404)
        
    config = memory_module.get("config", {}).copy()
    config["save_confidence_threshold"] = save_confidence_threshold
    
    module_manager.update_module_config("memory", config)
    
    return Response(status_code=200, headers={"HX-Trigger": json.dumps({"showMessage": {"level": "success", "message": "Persistence settings saved"}})})