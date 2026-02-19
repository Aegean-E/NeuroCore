import httpx

from core.settings import settings


class LLMBridge:
    def __init__(self, base_url: str, api_key: str = None):
        # Normalize base_url: strip trailing slashes
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    def _get_url(self, path: str):
        """Constructs a full URL for the given path."""
        # The base_url from settings should be the complete base, e.g., "http://host:port/v1"
        # The path is e.g., "/chat/completions"
        return f"{self.base_url}/{path.lstrip('/')}"

    async def chat_completion(self, messages, model: str = None, temperature: float = None):
        """Sends a chat completion request to the LLM API."""
        url = self._get_url("/chat/completions")
        
        # Use settings as a fallback for model and temperature
        final_model = model or settings.get("default_model")
        final_temperature = temperature if temperature is not None else settings.get("temperature")
        max_tokens = settings.get("max_tokens")

        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": final_model,
            "messages": messages,
            "temperature": final_temperature,
            "max_tokens": max_tokens,
            "stream": False
        }
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                response = await client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                return response.json()
            except Exception as e:
                return {"error": str(e)}

    async def get_models(self):
        """Fetches available models from the LLM API."""
        url = self._get_url("/models")
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(url, headers=headers)
                response.raise_for_status()
                return response.json()
            except Exception as e:
                return {"error": str(e)}
