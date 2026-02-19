import json
from fastapi import APIRouter, Request, Form, Depends, HTTPException, Response
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from core.settings import SettingsManager, settings
from core.dependencies import get_settings_manager, get_module_manager, get_llm_bridge
from core.module_manager import ModuleManager
from core.flow_manager import flow_manager
from core.llm import LLMBridge

router = APIRouter()
templates = Jinja2Templates(directory="web/templates")

# --- System & Navigation ---

@router.get("/", response_class=HTMLResponse)
async def read_root(request: Request, module_manager: ModuleManager = Depends(get_module_manager)):
    return templates.TemplateResponse(request, "index.html", {
        "modules": module_manager.get_all_modules(),
        "active_module": None
    })

@router.get("/navbar", response_class=HTMLResponse)
async def get_navbar(request: Request, module_manager: ModuleManager = Depends(get_module_manager)):
    return templates.TemplateResponse(request, "navbar.html", {"modules": module_manager.get_all_modules()})

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
async def get_module_details(request: Request, module_id: str, module_manager: ModuleManager = Depends(get_module_manager)):
    module = module_manager.modules.get(module_id)
    if not module:
        raise HTTPException(status_code=404, detail="Module not found")
    
    config_display = module.get('config', {}).copy()
    if module_id == 'memory':
        for key in ['save_default_confidence', 'save_confidence_threshold', 'recall_limit', 'recall_min_score', 'consolidation_threshold', 'auto_consolidation_hours', 'arbiter_model', 'arbiter_prompt']:
            config_display.pop(key, None)
    elif module_id == 'llm_module':
        for key in ['temperature', 'max_tokens']:
            config_display.pop(key, None)
    elif module_id == 'chat':
        for key in ['auto_rename_turns']:
            config_display.pop(key, None)
    elif module_id == 'telegram':
        for key in ['bot_token', 'chat_id']:
            config_display.pop(key, None)
        
    formatted_config = json.dumps(config_display, indent=4)
    
    # Check if there are any keys left to display
    has_visible_config = len(config_display) > 0
    
    return templates.TemplateResponse(request, "module_details.html", {
        "module": module, 
        "formatted_config": formatted_config,
        "has_visible_config": has_visible_config
    })

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

@router.post("/modules/{module_id}/config")
async def save_module_config(request: Request, module_id: str, config_json: str = Form(...), module_manager: ModuleManager = Depends(get_module_manager)):
    try:
        new_config = json.loads(config_json)
        
        # Preserve hidden keys for memory module
        if module_id == 'memory':
            current_module = module_manager.modules.get(module_id)
            if current_module:
                current_config = current_module.get('config', {})
                for key in ['save_default_confidence', 'save_confidence_threshold', 'recall_limit', 'recall_min_score', 'consolidation_threshold', 'auto_consolidation_hours', 'arbiter_model', 'arbiter_prompt']:
                    if key in current_config:
                        new_config[key] = current_config[key]
        elif module_id == 'llm_module':
            current_module = module_manager.modules.get(module_id)
            if current_module:
                current_config = current_module.get('config', {})
                for key in ['temperature', 'max_tokens']:
                    if key in current_config:
                        new_config[key] = current_config[key]
        elif module_id == 'chat':
            current_module = module_manager.modules.get(module_id)
            if current_module:
                current_config = current_module.get('config', {})
                for key in ['auto_rename_turns']:
                    if key in current_config:
                        new_config[key] = current_config[key]
        elif module_id == 'telegram':
            current_module = module_manager.modules.get(module_id)
            if current_module:
                current_config = current_module.get('config', {})
                for key in ['bot_token', 'chat_id']:
                    if key in current_config:
                        new_config[key] = current_config[key]
                    
        module_manager.update_module_config(module_id, new_config)
        return Response(status_code=200, headers={"HX-Trigger": json.dumps({"showMessage": {"level": "success", "message": "Configuration saved"}})})
    except json.JSONDecodeError:
        return Response(status_code=400, headers={"HX-Trigger": json.dumps({"showMessage": {"level": "error", "message": "Invalid JSON format"}})})
    except Exception as e:
        return Response(status_code=500, headers={"HX-Trigger": json.dumps({"showMessage": {"level": "error", "message": str(e)}})})


# --- AI Flow ---

@router.get("/ai-flow", response_class=HTMLResponse)
async def ai_flow_page(request: Request, module_manager: ModuleManager = Depends(get_module_manager)):
    active_flow_id = settings.get("active_ai_flow")
    return templates.TemplateResponse(request, "ai_flow.html", {
        "modules": module_manager.get_all_modules(),
        "flows": flow_manager.list_flows(),
        "active_flow_id": active_flow_id
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
    flow_manager.save_flow(name=name, nodes=json.loads(nodes), connections=json.loads(connections), flow_id=flow_id)
    
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
        "settings": settings_man.settings, "modules": module_manager.get_all_modules()
    })

@router.post("/settings/save")
async def save_settings_route(llm_api_url: str = Form(...), llm_api_key: str = Form(""), embedding_api_url: str = Form(""), default_model: str = Form(...), embedding_model: str = Form(""), settings_man: SettingsManager = Depends(get_settings_manager)):
    settings_man.save_settings({
        "llm_api_url": llm_api_url, "llm_api_key": llm_api_key, "embedding_api_url": embedding_api_url,
        "default_model": default_model, "embedding_model": embedding_model
    })
    return Response(status_code=200, headers={"HX-Trigger": json.dumps({"showMessage": {"level": "success", "message": "Settings saved successfully"}})})