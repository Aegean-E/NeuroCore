import httpx
import json
import os

class LMStudioBridge:
    def __init__(self, base_url="http://localhost:1234/v1"):
        self.base_url = base_url

    async def chat_completion(self, messages, model="local-model", temperature=0.7):
        """Sends a chat completion request to LM Studio."""
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
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
        """Fetches available models from LM Studio."""
        url = f"{self.base_url}/models"
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(url)
                return response.json()
            except Exception as e:
                return {"error": str(e)}

# Global instance
llm = LMStudioBridge()
