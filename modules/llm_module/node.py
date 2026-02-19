import os
import json
from core.llm import LLMBridge
from core.settings import settings

class LLMExecutor:
    def __init__(self):
        self.module_config = {}
        try:
            module_path = os.path.join(os.path.dirname(__file__), "module.json")
            if os.path.exists(module_path):
                with open(module_path, "r") as f:
                    self.module_config = json.load(f).get("config", {})
        except Exception as e:
            print(f"Error loading llm_module config: {e}")

    async def receive(self, input_data: dict, config: dict = None) -> dict:
        config = config or {}
        """
        Receives data from an upstream node (like Chat Input),
        and executes the core logic of this node (calling the LLM).
        """
        llm_bridge = LLMBridge(
            base_url=settings.get("llm_api_url"),
            api_key=settings.get("llm_api_key")
        )
        
        messages = input_data.get("messages")
        if not messages:
            return {"error": "No 'messages' field provided in input_data for llm_module."}

        model = config.get("model") or input_data.get("model")
        
        temperature = float(config.get("temperature") or input_data.get("temperature") or self.module_config.get("temperature", 0.7))
        max_tokens = int(config.get("max_tokens") or input_data.get("max_tokens") or self.module_config.get("max_tokens", 2048))
        
        # Return the result of the core logic
        return await llm_bridge.chat_completion(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens
        )

    async def send(self, processed_data: dict) -> dict:
        """Sends the processed data (the LLM response) to the next node."""
        return processed_data

async def get_executor_class(node_type_id: str):
    if node_type_id == "llm_module":
        return LLMExecutor
    return None