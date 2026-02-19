from core.llm import LLMBridge
from core.settings import settings

class LLMExecutor:
    async def receive(self, input_data: dict) -> dict:
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

        model = input_data.get("model")
        temperature = input_data.get("temperature")
        
        # Return the result of the core logic
        return await llm_bridge.chat_completion(
            messages=messages,
            model=model,
            temperature=temperature
        )

    async def send(self, processed_data: dict) -> dict:
        """Sends the processed data (the LLM response) to the next node."""
        return processed_data

async def get_executor_class(node_type_id: str):
    if node_type_id == "llm_module":
        return LLMExecutor
    return None