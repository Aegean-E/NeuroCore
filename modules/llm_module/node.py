import json
import os
from core.llm import LLMBridge
from core.settings import settings

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
            print(f"Error loading llm config: {e}")
        return cls._cache["data"]

class LLMExecutor:
    def __init__(self):
        self.module_config = ConfigLoader.get_config()

    async def receive(self, input_data: dict, config: dict = None) -> dict:
        if not input_data:
            return {"error": "No input data received"}
        
        messages = input_data.get("messages", [])
        if not messages:
            return {"error": "No 'messages' field provided in input_data for llm_module."}

        # Merge configs: Input > Node Config > Module Config > Settings (handled by LLMBridge)
        config = config or {}
        
        # Extract tools
        tools = input_data.get("tools") or config.get("tools")
        
        # Initialize Bridge
        llm = LLMBridge(
            base_url=settings.get("llm_api_url"),
            api_key=settings.get("llm_api_key")
        )

        # Parameters
        model = config.get("model") or self.module_config.get("default_model")
        temperature = config.get("temperature")
        if temperature is None:
            temperature = self.module_config.get("temperature")
        
        max_tokens = config.get("max_tokens")
        if max_tokens is None:
            max_tokens = self.module_config.get("max_tokens")

        response = await llm.chat_completion(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            tool_choice="auto" if tools else None
        )

        # Pass through messages and tools for continuity in flows (e.g. Tool Dispatcher loops)
        if isinstance(response, dict):
            # If the response is an error, we still might want context, but usually we stop.
            if "error" not in response:
                response["messages"] = messages
                if tools:
                    response["tools"] = tools
                
                # Also pass through available_tools if present (from System Prompt)
                if "available_tools" in input_data:
                    response["available_tools"] = input_data["available_tools"]

        return response

    async def send(self, processed_data: dict) -> dict:
        return processed_data

async def get_executor_class(node_type_id: str):
    if node_type_id == "llm_module":
        return LLMExecutor
    return None