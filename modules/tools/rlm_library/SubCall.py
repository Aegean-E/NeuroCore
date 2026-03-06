# sub_call.py - Recursive LLM call
# Used in RLM (Recursive Language Model) to make sub-calls on chunks of the prompt
# This is THE key feature that enables processing arbitrarily long inputs

import asyncio
import json

args = args.get("args", {})
repl_state = args.get("_repl_state", {})

prompt = args.get("prompt", "")
model = args.get("model")  # Optional: use specific model for sub-call
max_tokens = args.get("max_tokens", 2000)

if not prompt:
    result = {"error": "No prompt provided for sub_call"}
else:
    # Get settings - we need to do this synchronously for sandbox
    try:
        # Note: In the sandbox we can't directly import settings
        # So we'll use environment variables or defaults
        import os
        
        # Try to get from environment or use defaults
        api_url = os.environ.get("LLM_API_URL", "http://localhost:11434/v1")
        api_key = os.environ.get("LLM_API_KEY", "not-needed")
        
        # If model not specified, try to get default
        if not model:
            model = os.environ.get("DEFAULT_MODEL", "llama3")
        
        # Use httpx for async HTTP call
        import httpx
        
        # Prepare the request
        headers = {
            "Content-Type": "application/json"
        }
        if api_key and api_key != "not-needed":
            headers["Authorization"] = f"Bearer {api_key}"
        
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens
        }
        
        # Make the API call synchronously
        # Note: In production, this should be async - but sandbox runs sync
        try:
            response = httpx.post(
                f"{api_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=60.0
            )
            
            if response.status_code == 200:
                data = response.json()
                if "choices" in data and len(data["choices"]) > 0:
                    content = data["choices"][0]["message"].get("content", "")
                    result = {
                        "success": True,
                        "response": content,
                        "model_used": model,
                        "response_length": len(content)
                    }
                else:
                    result = {"error": "No response content from sub_call"}
            else:
                result = {"error": f"API error: {response.status_code}"}
                
        except Exception as e:
            # Fallback: if we can't make HTTP call, simulate response
            result = {
                "error": f"sub_call requires LLM API access: {str(e)}",
                "note": "Configure LLM_API_URL and LLM_API_KEY in settings"
            }
            
    except Exception as e:
        result = {"error": f"sub_call failed: {str(e)}"}

