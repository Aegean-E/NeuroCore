from fastapi import APIRouter, Request, Form
from fastapi.responses import Response
import json
from .service import discord_service

router = APIRouter()


@router.post("/settings/save")
async def save_discord_settings(
    request: Request,
    bot_token: str = Form(...),
    channel_id: str = Form(...),
):
    module_manager = request.app.state.module_manager
    discord_module = module_manager.modules.get("discord")
    if not discord_module:
        return Response(status_code=404)

    config = discord_module.get("config", {}).copy()
    config["bot_token"] = bot_token
    config["channel_id"] = channel_id

    module_manager.update_module_config("discord", config)

    return Response(
        status_code=200,
        headers={
            "HX-Trigger": json.dumps({
                "showMessage": {
                    "level": "success",
                    "message": "Discord settings saved",
                }
            })
        },
    )


# Start the service when the router is loaded (i.e., the module is enabled).
discord_service.start()
