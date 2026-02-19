from fastapi import APIRouter, Request, Form
from fastapi.responses import Response
import json
from .service import telegram_service

router = APIRouter()

@router.post("/settings/save")
async def save_telegram_settings(request: Request, bot_token: str = Form(...), chat_id: int = Form(...)):
    module_manager = request.app.state.module_manager
    telegram_module = module_manager.modules.get("telegram")
    if not telegram_module:
        return Response(status_code=404)
        
    config = telegram_module.get("config", {}).copy()
    config["bot_token"] = bot_token
    config["chat_id"] = chat_id
    
    module_manager.update_module_config("telegram", config)
    
    return Response(status_code=200, headers={"HX-Trigger": json.dumps({"showMessage": {"level": "success", "message": "Telegram settings saved"}})})

# Start the service when the router is loaded (module enabled)
telegram_service.start()