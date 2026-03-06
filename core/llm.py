import httpx
import logging
import threading

from core.settings import settings

logger = logging.getLogger(__name__)

# Module-level shared client for connection pooling
_shared_client: httpx.AsyncClient = None
_client_lock = threading.Lock()


async def get_shared_client(timeout: float = 60.0) -> httpx.AsyncClient:
    """Get or create a shared AsyncClient for connection pooling."""
    global _shared_client, _client_lock
    
    if _shared_client is None:
        _shared_client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout),
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=100)
        )
    return _shared_client


async def close_shared_client():
    """Close the shared client. Call on application shutdown."""
    global _shared_client
    if _shared_client is not None:
        await _shared_client.aclose()
        _shared_client = None


class LLMBridge:
    def __init__(self, base_url: str, api_key: str = None, embedding_base_url: str = None, embedding_model: str = None, client: httpx.AsyncClient = None, timeout: float = 60.0):
        # Normalize base_url: strip trailing slashes
        self.base_url = base_url.rstrip("/") if base_url else ""
        self.embedding_base_url = embedding_base_url.rstrip("/") if embedding_base_url else self.base_url
        self.api_key = api_key
        self.embedding_model = embedding_model
        self.client = client
        self.timeout = timeout

    def _get_url(self, path: str, use_embedding_url: bool = False):
        """Constructs a full URL for the given path."""
        # The base_url from settings should be the complete base, e.g., "http://host:port/v1"
        # The path is e.g., "/chat/completions"
        base = self.embedding_base_url if use_embedding_url else self.base_url
        return f"{base}/{path.lstrip('/')}"

    async def chat_completion(self, messages, model: str = None, temperature: float = None, max_tokens: int = None, tools: list = None, tool_choice: str = None, response_format: dict = None):
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
        if response_format:
            payload["response_format"] = response_format
        
        try:
            # Use injected client, shared client, or create new one
            client_to_use = self.client
            if not client_to_use:
                client_to_use = await get_shared_client(self.timeout)
            
            if client_to_use:
                response = await client_to_use.post(url, json=payload, headers=headers, timeout=self.timeout)
                response.raise_for_status()
                return response.json()
            else:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(url, json=payload, headers=headers)
                    response.raise_for_status()
                    return response.json()
        except Exception as e:
            logger.error(f"Chat completion error: {e}")
            return {"error": str(e)}

    async def get_models(self):
        """Fetches available models from the LLM API."""
        url = self._get_url("/models")
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            # Use injected client, shared client, or create new one
            client_to_use = self.client
            if not client_to_use:
                client_to_use = await get_shared_client(self.timeout)
            
            if client_to_use:
                response = await client_to_use.get(url, headers=headers)
                response.raise_for_status()
                return response.json()
            else:
                async with httpx.AsyncClient() as client:
                    response = await client.get(url, headers=headers)
                    response.raise_for_status()
                    return response.json()
        except Exception as e:
            logger.error(f"Get models error: {e}")
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
            # Use injected client, shared client, or create new one
            client_to_use = self.client
            if not client_to_use:
                client_to_use = await get_shared_client(self.timeout)
            
            # FIX: Use self.timeout instead of hardcoded 30.0
            if client_to_use:
                response = await client_to_use.post(url, json=payload, headers=headers, timeout=self.timeout)
            else:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(url, json=payload, headers=headers)
            
            response.raise_for_status()
            data = response.json()
            # Standard OpenAI format: data['data'][0]['embedding']
            if "data" in data and len(data["data"]) > 0:
                return data["data"][0]["embedding"]
            return None
        except Exception as e:
            # FIX: Use logger instead of print
            logger.error(f"Embedding error: {e}")
            return None
