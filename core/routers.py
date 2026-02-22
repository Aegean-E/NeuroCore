import json
from datetime import datetime
from fastapi import APIRouter, Request, Form, Depends, HTTPException, Response, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from core.settings import SettingsManager, settings
from core.dependencies import get_settings_manager, get_module_manager, get_llm_bridge
from core.module_manager import ModuleManager
from core.flow_manager import flow_manager
from core.llm import LLMBridge
from core.debug import debug_logger

router = APIRouter()
templates = Jinja2Templates(directory="web/templates")

# Centralized definition of config keys that should be hidden from the generic JSON editor
HIDDEN_CONFIG_KEYS = {
    'memory': ['save_default_confidence', 'save_confidence_threshold', 'recall_limit', 'recall_min_score', 'consolidation_threshold', 'auto_consolidation_hours', 'arbiter_model', 'arbiter_prompt'],
    'llm_module': ['temperature', 'max_tokens'],
    'chat': ['auto_rename_turns'],
    'telegram': ['bot_token', 'chat_id']
}

# --- System & Navigation ---

@router.get("/", response_class=HTMLResponse)
async def read_root(request: Request, module_manager: ModuleManager = Depends(get_module_manager), settings_man: SettingsManager = Depends(get_settings_manager)):
    return templates.TemplateResponse(request, "index.html", {
        "modules": module_manager.get_all_modules(),
        "active_module": None,
        "settings": settings_man.settings
    })

@router.get("/navbar", response_class=HTMLResponse)
async def get_navbar(request: Request, module_manager: ModuleManager = Depends(get_module_manager), settings_man: SettingsManager = Depends(get_settings_manager)):
    return templates.TemplateResponse(request, "navbar.html", {"modules": module_manager.get_all_modules(), "settings": settings_man.settings})

@router.get("/llm-status", response_class=HTMLResponse)
async def get_llm_status(request: Request, llm: LLMBridge = Depends(get_llm_bridge)):
    api_status_check = await llm.get_models()
    api_online = "error" not in api_status_check
    return f"""
    <div class="flex items-center space-x-2">
        <div class="w-2 h-2 rounded-full {"bg-emerald-500 animate-pulse" if api_online else "bg-red-500"}"></div>
        <span class="text-xs {"text-slate-400" if api_online else "text-red-400"}">LLM Status: {"Online" if api_online else "Offline"}</span>
    </div>
    """

# --- Module Management ---

@router.get("/modules/list", response_class=HTMLResponse)
async def list_modules(request: Request, module_manager: ModuleManager = Depends(get_module_manager)):
    return templates.TemplateResponse(request, "module_list.html", {"modules": module_manager.get_all_modules()})

@router.get("/modules/{module_id}/details", response_class=HTMLResponse)
async def get_module_details(request: Request, module_id: str, module_manager: ModuleManager = Depends(get_module_manager), settings_man: SettingsManager = Depends(get_settings_manager)):
    module = module_manager.modules.get(module_id)
    if not module:
        raise HTTPException(status_code=404, detail="Module not found")
    
    config_display = module.get('config', {}).copy()
    keys_to_hide = HIDDEN_CONFIG_KEYS.get(module_id, [])
    for key in keys_to_hide:
        config_display.pop(key, None)
        
    formatted_config = json.dumps(config_display, indent=4)
    
    # Check if there are any keys left to display
    has_visible_config = len(config_display) > 0
    
    return templates.TemplateResponse(request, "module_details.html", {
        "module": module, 
        "formatted_config": formatted_config,
        "has_visible_config": has_visible_config,
        "settings": settings_man.settings
    })

@router.post("/modules/{module_id}/config")
async def save_module_config(request: Request, module_id: str, module_manager: ModuleManager = Depends(get_module_manager)):
    try:
        form_data = await request.form()
        
        # Handle system_prompt module specially
        if module_id == "system_prompt":
            new_config = {}
            if "system_prompt" in form_data:
                new_config["system_prompt"] = form_data["system_prompt"]
            if "enabled_tools" in form_data:
                try:
                    new_config["enabled_tools"] = json.loads(form_data["enabled_tools"])
                except json.JSONDecodeError:
                    new_config["enabled_tools"] = []
        else:
            # Standard JSON config
            if "config_json" not in form_data:
                return Response(status_code=400, headers={"HX-Trigger": json.dumps({"showMessage": {"level": "error", "message": "Missing config_json field"}})})
            
            new_config = json.loads(form_data["config_json"])
        
        # Preserve hidden keys by merging from existing config
        keys_to_preserve = HIDDEN_CONFIG_KEYS.get(module_id, [])
        current_module = module_manager.modules.get(module_id)
        
        if current_module:
            current_config = current_module.get('config', {})
            for key in keys_to_preserve:
                if key in current_config:
                    new_config[key] = current_config[key]
                    
        module_manager.update_module_config(module_id, new_config)
        return Response(status_code=200, headers={"HX-Trigger": json.dumps({"showMessage": {"level": "success", "message": "Configuration saved"}})})
    except json.JSONDecodeError:
        return Response(status_code=400, headers={"HX-Trigger": json.dumps({"showMessage": {"level": "error", "message": "Invalid JSON format"}})})
    except Exception as e:
        return Response(status_code=500, headers={"HX-Trigger": json.dumps({"showMessage": {"level": "error", "message": str(e)}})})

@router.post("/modules/reorder")
async def reorder_modules(request: Request, order: str = Form(...), module_manager: ModuleManager = Depends(get_module_manager)):
    module_ids = order.split(',')
    try:
        module_manager.reorder_modules(module_ids)
        # Trigger a refresh of the navbar to reflect the new order
        return Response(status_code=200, headers={"HX-Trigger": "modulesChanged"})
    except Exception as e:
        return Response(status_code=500, headers={"HX-Trigger": json.dumps({"showMessage": {"level": "error", "message": f"Failed to reorder: {e}"}})})

@router.post("/modules/{module_id}/{action}")
async def toggle_module(request: Request, module_id: str, action: str, module_manager: ModuleManager = Depends(get_module_manager)):
    if action == "enable":
        module = module_manager.enable_module(module_id)
    elif action == "disable":
        module = module_manager.disable_module(module_id)
    else:
        raise HTTPException(status_code=400, detail="Invalid action")
    
    if not module:
        raise HTTPException(status_code=404, detail="Module not found")
    
    formatted_config = json.dumps(module.get('config', {}), indent=4)
    return templates.TemplateResponse(
        request, "module_details.html", {"module": module, "formatted_config": formatted_config}, headers={"HX-Trigger": json.dumps({"modulesChanged": None, "showMessage": {"level": "success", "message": f"Module {action}d"}})}
    )


# --- AI Flow ---

@router.get("/ai-flow", response_class=HTMLResponse)
async def ai_flow_page(request: Request, module_manager: ModuleManager = Depends(get_module_manager), settings_man: SettingsManager = Depends(get_settings_manager)):
    active_flow_id = settings_man.get("active_ai_flow")
    return templates.TemplateResponse(request, "ai_flow.html", {
        "modules": module_manager.get_all_modules(),
        "flows": flow_manager.list_flows(),
        "active_flow_id": active_flow_id,
        "settings": settings_man.settings
    })

@router.get("/ai-flow/{flow_id}", response_class=JSONResponse)
async def get_flow_data(flow_id: str):
    flow = flow_manager.get_flow(flow_id)
    if not flow:
        raise HTTPException(status_code=404, detail="Flow not found")
    return flow

@router.post("/ai-flow/save")
async def save_ai_flow(request: Request, name: str = Form(...), nodes: str = Form(...), connections: str = Form(...), flow_id: str = Form(None)):
    if not flow_id:
        flow_id = None
    
    try:
        flow_manager.save_flow(name=name, nodes=json.loads(nodes), connections=json.loads(connections), flow_id=flow_id)
    except json.JSONDecodeError:
        return Response(status_code=400, headers={"HX-Trigger": json.dumps({"showMessage": {"level": "error", "message": "Invalid JSON in flow data"}})})
    
    return templates.TemplateResponse(request, "ai_flow_list.html", {
        "flows": flow_manager.list_flows(),
        "active_flow_id": settings.get("active_ai_flow")
    }, headers={"HX-Trigger": json.dumps({"showMessage": {"level": "success", "message": "Flow saved successfully"}})})

@router.post("/ai-flow/{flow_id}/rename", response_class=HTMLResponse)
async def rename_flow(request: Request, flow_id: str, name: str = Form(...), settings_man: SettingsManager = Depends(get_settings_manager)):
    flow_manager.rename_flow(flow_id, name)
    return templates.TemplateResponse(request, "ai_flow_list.html", {
        "flows": flow_manager.list_flows(),
        "active_flow_id": settings_man.get("active_ai_flow")
    })

@router.post("/ai-flow/{flow_id}/set-active", response_class=HTMLResponse)
async def set_active_flow(request: Request, flow_id: str, settings_man: SettingsManager = Depends(get_settings_manager)):
    settings_man.save_settings({"active_ai_flow": flow_id})
    return templates.TemplateResponse(request, "ai_flow_list.html", {
        "flows": flow_manager.list_flows(),
        "active_flow_id": flow_id
    })

@router.post("/ai-flow/{flow_id}/delete", response_class=HTMLResponse)
async def delete_flow(request: Request, flow_id: str, settings_man: SettingsManager = Depends(get_settings_manager)):
    if settings_man.get("active_ai_flow") == flow_id:
        settings_man.save_settings({"active_ai_flow": None})
    flow_manager.delete_flow(flow_id)
    return templates.TemplateResponse(request, "ai_flow_list.html", {
        "flows": flow_manager.list_flows(),
        "active_flow_id": settings_man.get("active_ai_flow")
    })

# --- Settings ---

@router.get("/settings", response_class=HTMLResponse)
async def get_settings(request: Request, settings_man: SettingsManager = Depends(get_settings_manager), module_manager: ModuleManager = Depends(get_module_manager)):
    return templates.TemplateResponse(request, "settings.html", {
        "settings": settings_man.settings, "modules": module_manager.get_all_modules(),
        "system_time": datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    })

@router.get("/system-time", response_class=HTMLResponse)
async def get_system_time(request: Request):
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")

@router.get("/settings/modules-nav", response_class=HTMLResponse)
async def get_settings_modules_nav(request: Request, module_manager: ModuleManager = Depends(get_module_manager)):
    return templates.TemplateResponse(request, "settings_modules_nav.html", {"modules": module_manager.get_all_modules()})

@router.get("/footer", response_class=HTMLResponse)
async def get_footer(request: Request, settings_man: SettingsManager = Depends(get_settings_manager)):
    return templates.TemplateResponse(request, "footer.html", {"settings": settings_man.settings})

@router.get("/settings/export/flows")
async def export_flows():
    """Downloads the current ai_flows.json file."""
    flows = flow_manager.list_flows()
    # Convert list back to dict format for the file, or just dump the list? 
    # The FlowManager loads a dict, so let's dump the internal dict structure if possible, 
    # but flow_manager.flows is available.
    data = flow_manager.flows
    return JSONResponse(
        content=data,
        headers={"Content-Disposition": 'attachment; filename="ai_flows_backup.json"'}
    )

@router.post("/settings/save")
async def save_settings_route(request: Request, settings_man: SettingsManager = Depends(get_settings_manager)):
    form_data = await request.form()
    updates = {}
    
    # Handle text fields (update only if present)
    text_fields = ["llm_api_url", "llm_api_key", "embedding_api_url", "default_model", "embedding_model"]
    for field in text_fields:
        if field in form_data:
            updates[field] = form_data[field]
            
    # Handle debug_mode checkbox (only if the form intended to submit it)
    if "save_debug_mode" in form_data:
        updates["debug_mode"] = form_data.get("debug_mode") == "on"
    
    if "save_ui_wide_mode" in form_data:
        updates["ui_wide_mode"] = form_data.get("ui_wide_mode") == "on"
        
    if "save_ui_show_footer" in form_data:
        updates["ui_show_footer"] = form_data.get("ui_show_footer") == "on"

    settings_man.save_settings(updates)
    return Response(status_code=200, headers={"HX-Trigger": json.dumps({"settingsChanged": None, "showMessage": {"level": "success", "message": "Settings saved successfully"}})})

# --- Debug ---

@router.get("/debug", response_class=HTMLResponse)
async def debug_page(request: Request, settings_man: SettingsManager = Depends(get_settings_manager), module_manager: ModuleManager = Depends(get_module_manager)):
    if not settings_man.get("debug_mode"):
        return RedirectResponse("/")
    return templates.TemplateResponse(request, "debug.html", {"settings": settings_man.settings, "modules": module_manager.get_all_modules()})

@router.get("/debug/logs", response_class=HTMLResponse)
async def get_debug_logs(request: Request):
    return templates.TemplateResponse(request, "debug_logs.html", {"logs": debug_logger.get_logs()})

@router.get("/debug/events", response_class=JSONResponse)
async def get_debug_events(request: Request, since: float = 0):
    return debug_logger.get_recent_logs(since)

@router.post("/debug/clear")
async def clear_debug_logs(request: Request):
    debug_logger.clear()
    return templates.TemplateResponse(request, "debug_logs.html", {"logs": []})