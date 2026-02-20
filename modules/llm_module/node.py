import os
import json
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
            print(f"Error loading llm_module config: {e}")
        return cls._cache["data"]

class LLMExecutor:
    def __init__(self):
        self.module_config = ConfigLoader.get_config()

    async def receive(self, input_data: dict, config: dict = None) -> dict:
        config = config or {}
        """
        Receives data from an upstream node (like System Prompt or Chat Input),
        and executes the core logic of this node (calling the LLM).
        Gets available tools from system prompt and knows how to call tool dispatcher.
        """
        llm_bridge = LLMBridge(
            base_url=settings.get("llm_api_url"),
            api_key=settings.get("llm_api_key")
        )
        
        messages = input_data.get("messages")
        if not messages:
            return {"error": "No 'messages' field provided in input_data for llm_module."}

        model = config.get("model") or input_data.get("model")
        
        # Helper to resolve priority: config > input_data > module_config > default
        # Handles 0 values correctly (unlike 'or' operator)
        def get_val(key, default, type_cast=None):
            val = config.get(key) if config else None
            if val is None:
                val = input_data.get(key) if input_data else None
            if val is None:
                val = self.module_config.get(key) if self.module_config else None
            
            if val is None:
                return default
                
            if type_cast:
                try:
                    return type_cast(val)
                except (ValueError, TypeError):
                    return default
            return val

        temperature = get_val("temperature", 0.7, float)
        max_tokens = get_val("max_tokens", 2048, int)
        
        # Priority: config > input_data (from system prompt) > module_config
        # Tools come from system prompt via input_data, or can be overridden in config
        tools = config.get("tools") or input_data.get("tools")
        tool_choice = config.get("tool_choice") or input_data.get("tool_choice", "auto" if tools else None)
        
        # Return the result of the core logic
        return await llm_bridge.chat_completion(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            tool_choice=tool_choice
        )

    async def send(self, processed_data: dict) -> dict:
        """Sends the processed data (the LLM response) to the next node."""
        return processed_data

async def get_executor_class(node_type_id: str):
    if node_type_id == "llm_module":
        return LLMExecutor
    return None