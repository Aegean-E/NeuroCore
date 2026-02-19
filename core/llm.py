import httpx

from core.settings import settings


class LLMBridge:
    def __init__(self, base_url: str):
        self.base_url = base_url

    async def chat_completion(self, messages, model: str = None, temperature: float = None):
        """Sends a chat completion request to the LLM API."""
        url = f"{self.base_url}/chat/completions"

        # Use settings as a fallback for model and temperature
        final_model = model or settings.get("default_model")
        final_temperature = temperature if temperature is not None else settings.get("temperature")
        max_tokens = settings.get("max_tokens")

        payload = {
            "model": final_model,
            "messages": messages,
            "temperature": final_temperature,
            "max_tokens": max_tokens,
            "stream": False
        }
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                return response.json()
            except Exception as e:
                return {"error": str(e)}

    async def get_models(self):
        """Fetches available models from the LLM API."""
        url = f"{self.base_url}/models"
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(url)
                return response.json()
            except Exception as e:
                return {"error": str(e)}
