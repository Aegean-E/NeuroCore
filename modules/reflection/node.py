import json
import re
from core.llm import LLMBridge
from core.settings import settings


class ReflectionExecutor:
    def __init__(self):
        self.llm = LLMBridge(
            base_url=settings.get("llm_api_url"),
            api_key=settings.get("llm_api_key")
        )

    def _extract_user_message(self, input_data: dict) -> str:
        """Extract the original user request."""
        messages = input_data.get("messages", [])
        
        for msg in reversed(messages):
            if isinstance(msg, dict) and msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    return content
        return ""

    def _extract_assistant_response(self, input_data: dict) -> str:
        """Extract the latest assistant response."""
        # Check content field first (from Agent Loop)
        content = input_data.get("content")
        if content:
            return content
        
        # Check messages
        messages = input_data.get("messages", [])
        for msg in reversed(messages):
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                return msg.get("content", "")
        
        return ""

    async def receive(self, input_data: dict, config: dict = None) -> dict:
        """
        Evaluates if the response satisfies the user's request.
        
        Input:
            - messages: conversation history
            - OR content: assistant response
            
        Output:
            - reflection.satisfied: true/false
            - reflection.reason: explanation
            - reflection.needs_improvement: what could be better
        """
        if input_data is None:
            input_data = {}
        
        config = config or {}
        
        # Extract user request and assistant response
        user_request = self._extract_user_message(input_data)
        assistant_response = self._extract_assistant_response(input_data)
        
        if not user_request or not assistant_response:
            # Not enough data to reflect
            result = input_data.copy()
            result["reflection"] = {
                "satisfied": True,
                "reason": "No content to evaluate",
                "needs_improvement": None
            }
            return result
        
        # Build reflection prompt
        reflection_prompt = config.get("default_reflection_prompt", 
            "You are a reflection agent. Evaluate if the response satisfies the request.\n"
            "Respond with JSON: {\"satisfied\": true/false, \"reason\": \"...\", \"needs_improvement\": \"...\" or null}")
        
        reflection_messages = [
            {"role": "system", "content": reflection_prompt},
            {"role": "user", "content": f"User Request: {user_request}\n\nAssistant Response: {assistant_response}"}
        ]
        
        try:
            response = await self.llm.chat_completion(
                messages=reflection_messages,
                model=settings.get("default_model"),
                temperature=0.1,
                max_tokens=300
            )
            
            # Parse response
            reflection_result = {
                "satisfied": False,
                "reason": "Could not evaluate",
                "needs_improvement": None
            }
            
            if response and "choices" in response:
                content = response["choices"][0]["message"].get("content", "")
                
                # Try to parse JSON
                try:
                    # Remove comments
                    content = re.sub(r'//.*', '', content)
                    content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
                    
                    parsed = json.loads(content.strip())
                    
                    if isinstance(parsed, dict):
                        reflection_result = {
                            "satisfied": bool(parsed.get("satisfied", False)),
                            "reason": str(parsed.get("reason", "")),
                            "needs_improvement": parsed.get("needs_improvement")
                        }
                except (json.JSONDecodeError, IndexError):
                    pass
            
            # Build result
            result = input_data.copy()
            result["reflection"] = reflection_result
            
            # Also add satisfied flag at top level for easy routing
            result["satisfied"] = reflection_result["satisfied"]
            
            return result
            
        except Exception as e:
            result = input_data.copy()
            result["reflection"] = {
                "satisfied": True,
                "reason": f"Reflection error: {str(e)}",
                "needs_improvement": None
            }
            result["satisfied"] = True
            return result

    async def send(self, processed_data: dict) -> dict:
        return processed_data


async def get_executor_class(node_type_id: str):
    if node_type_id == "reflection":
        return ReflectionExecutor
    return None
