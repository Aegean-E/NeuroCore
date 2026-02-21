import json
import os
from .bridge import TelegramBridge

class ConfigLoader:
    @staticmethod
    def get_config():
        try:
            with open("modules/telegram/module.json", "r") as f:
                return json.load(f).get("config", {})
        except:
            return {}

    @staticmethod
    def is_enabled():
        try:
            with open("modules/telegram/module.json", "r") as f:
                return json.load(f).get("enabled", False)
        except:
            return False

class TelegramInputExecutor:
    async def receive(self, input_data: dict, config: dict = None) -> dict:
        # Ignore if triggered by Repeater
        if input_data.get("_repeat_count", 0) > 0:
            return None
        return input_data

    async def send(self, processed_data: dict) -> dict:
        # Double-check: ignore repeats here too to prevent error generation
        if processed_data and processed_data.get("_repeat_count", 0) > 0:
            return processed_data
        if "messages" not in processed_data and "text" not in processed_data:
             return {"error": "Flow started without 'messages'. 'Telegram Input' node requires it."}
        return processed_data

class TelegramOutputExecutor:
    async def receive(self, input_data: dict, config: dict = None) -> dict:
        if not input_data or "content" not in input_data:
            return input_data
            
        mod_config = ConfigLoader.get_config()
        bot_token = mod_config.get("bot_token")
        chat_id = mod_config.get("chat_id")
        
        if bot_token and chat_id:
            bridge = TelegramBridge(bot_token, chat_id)
            bridge.send_message(input_data["content"])
            
        return input_data

    async def send(self, processed_data: dict) -> dict:
        return processed_data

async def get_executor_class(node_type_id: str):
    if node_type_id == 'telegram_input':
        return TelegramInputExecutor
    if node_type_id == 'telegram_output':
        return TelegramOutputExecutor
    return None