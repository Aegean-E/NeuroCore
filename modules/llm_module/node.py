from core.llm import LLMBridge
from core.settings import settings

class ConfigLoader:
    @staticmethod
    def get_config():
        return settings.settings

class LLMExecutor:
    def __init__(self):
        self.bridge = LLMBridge(
            base_url=settings.get("llm_api_url"),
            api_key=settings.get("llm_api_key")
        )

    async def receive(self, input_data: dict, config: dict = None) -> dict:
        """
        Executes LLM completion.
        Precedence: Node Config > Input Data > Module Defaults
        """
        if config is None:
            config = {}
            
        module_defaults = ConfigLoader.get_config()
        
        # 1. Start with Module Defaults
        final_params = {
            "model": module_defaults.get("default_model"),
            "temperature": module_defaults.get("temperature"),
            "max_tokens": module_defaults.get("max_tokens")
        }
        
        # 2. Override with Input Data (only specific keys)
        for key in ["model", "temperature", "max_tokens", "tools", "tool_choice"]:
            if key in input_data:
                final_params[key] = input_data[key]
                
        # 3. Override with Node Config
        for key in ["model", "temperature", "max_tokens", "tools", "tool_choice"]:
            if key in config:
                final_params[key] = config[key]

        messages = input_data.get("messages", [])
        
        return await self.bridge.chat_completion(
            messages=messages,
            **final_params
        )

    async def send(self, processed_data: dict) -> dict:
        return processed_data

async def get_executor_class(node_type_id: str):
    if node_type_id == 'llm_module':
        return LLMExecutor
    return None