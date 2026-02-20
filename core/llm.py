import httpx

from core.settings import settings


class LLMBridge:
    def __init__(self, base_url: str, api_key: str = None, embedding_base_url: str = None, embedding_model: str = None, client: httpx.AsyncClient = None):
        # Normalize base_url: strip trailing slashes
        self.base_url = base_url.rstrip("/") if base_url else ""
        self.embedding_base_url = embedding_base_url.rstrip("/") if embedding_base_url else self.base_url
        self.api_key = api_key
        self.embedding_model = embedding_model
        self.client = client

    def _get_url(self, path: str, use_embedding_url: bool = False):
        """Constructs a full URL for the given path."""
        # The base_url from settings should be the complete base, e.g., "http://host:port/v1"
        # The path is e.g., "/chat/completions"
        base = self.embedding_base_url if use_embedding_url else self.base_url
        return f"{base}/{path.lstrip('/')}"

    async def chat_completion(self, messages, model: str = None, temperature: float = None, max_tokens: int = None, tools: list = None, tool_choice: str = None):
        """Sends a chat completion request to the LLM API."""
        url = self._get_url("/chat/completions")
        
        # Use settings as a fallback for model and temperature
        final_model = model or settings.get("default_model")
        final_temperature = temperature if temperature is not None else settings.get("temperature", 0.7)
        final_max_tokens = max_tokens if max_tokens is not None else settings.get("max_tokens", 2048)

        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": final_model,
            "messages": messages,
            "temperature": final_temperature,
            "max_tokens": final_max_tokens,
            "stream": False
        }

        if tools:
            payload["tools"] = tools
        if tool_choice:
            payload["tool_choice"] = tool_choice
        
        try:
            if self.client:
                response = await self.client.post(url, json=payload, headers=headers, timeout=60.0)
                response.raise_for_status()
                return response.json()
            else:
                async with httpx.AsyncClient(timeout=60.0) as client:
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

        try:
            if self.client:
                response = await self.client.get(url, headers=headers)
                response.raise_for_status()
                return response.json()
            else:
                async with httpx.AsyncClient() as client:
                    response = await client.get(url, headers=headers)
                    response.raise_for_status()
                    return response.json()
        except Exception as e:
            return {"error": str(e)}

    async def get_embedding(self, text: str, model: str = None):
        """Generates an embedding vector for the given text."""
        url = self._get_url("/embeddings", use_embedding_url=True)
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        # Some local servers use 'text-embedding-nomic-embed-text-v1.5' or similar
        payload = {
            "input": text,
            "model": model or self.embedding_model or settings.get("default_model") 
        }

        try:
            if self.client:
                response = await self.client.post(url, json=payload, headers=headers, timeout=30.0)
            else:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.post(url, json=payload, headers=headers)
            
            response.raise_for_status()
            data = response.json()
            # Standard OpenAI format: data['data'][0]['embedding']
            if "data" in data and len(data["data"]) > 0:
                return data["data"][0]["embedding"]
            return None
        except Exception as e:
            print(f"Embedding error: {e}")
            return None
