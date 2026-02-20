import os
import json
import threading
from .bridge import TelegramBridge

class ConfigLoader:
    _cache = {"mtime": 0, "data": {}}
    _path = os.path.join(os.path.dirname(__file__), "module.json")

    @classmethod
    def get_config(cls):
        try:
            if os.path.exists(cls._path):
                mtime = os.path.getmtime(cls._path)
                if mtime > cls._cache["mtime"]:
                    with open(cls._path, "r") as f:
                        cls._cache["data"] = json.load(f).get("config", {})
                    cls._cache["mtime"] = mtime
        except Exception as e:
            print(f"Error loading telegram config: {e}")
        return cls._cache["data"]

    @classmethod
    def is_enabled(cls):
        try:
            if os.path.exists(cls._path):
                with open(cls._path, "r") as f:
                    data = json.load(f)
                    return data.get("enabled", False)
        except:
            return False
        return False

class TelegramInputExecutor:
    async def receive(self, input_data: dict, config: dict = None) -> dict:
        return input_data

    async def send(self, processed_data: dict) -> dict:
        if "messages" not in processed_data:
             return {"error": "Flow started without 'messages'. 'Telegram Input' node requires it."}
        return processed_data

class TelegramOutputExecutor:
    def __init__(self):
        self.config = ConfigLoader.get_config()
        self.bot_token = self.config.get("bot_token")
        self.chat_id = self.config.get("chat_id")

    async def receive(self, input_data: dict, config: dict = None) -> dict:
        if not self.bot_token or not self.chat_id:
            print("Telegram Output: Missing bot_token or chat_id configuration.")
            return input_data

        text_to_send = None
        if "content" in input_data:
            text_to_send = input_data["content"]
        elif "choices" in input_data:
             try:
                text_to_send = input_data["choices"][0]["message"]["content"]
             except: pass
        
        if text_to_send:
            with TelegramBridge(self.bot_token, self.chat_id) as bridge:
                bridge.send_message(text_to_send)
            
        return input_data

    async def send(self, processed_data: dict) -> dict:
        return processed_data

async def get_executor_class(node_type_id: str):
    if node_type_id == "telegram_input":
        return TelegramInputExecutor
    if node_type_id == "telegram_output":
        return TelegramOutputExecutor
    return None