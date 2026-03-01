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
        """Extract the original user request from messages."""
        messages = input_data.get("messages", [])
        for msg in reversed(messages):
            if isinstance(msg, dict) and msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    return content
        return ""

    def _extract_assistant_response(self, input_data: dict) -> str:
        """Extract the latest assistant response.

        Checks the top-level 'content' key first (set by agent_loop),
        then falls back to scanning messages.
        """
        content = input_data.get("content")
        if content:
            return content

        messages = input_data.get("messages", [])
        for msg in reversed(messages):
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                return msg.get("content", "")
        return ""

    def _inject_improvement_message(self, input_data: dict, needs_improvement: str) -> dict:
        """
        Inject an improvement feedback message into the conversation so that
        when agent_loop receives the data again (via flow loop-back), it sees
        the critique and can produce a better response.

        The injected message has role='user' and is appended to 'messages'.
        """
        result = input_data.copy()
        messages = list(result.get("messages", []))
        messages.append({
            "role": "user",
            "content": (
                f"Your previous response needs improvement: {needs_improvement}\n"
                "Please try again with a better response."
            )
        })
        result["messages"] = messages
        return result

    async def receive(self, input_data: dict, config: dict = None) -> dict:
        """
        Evaluates whether the agent's response satisfies the user's request.

        When used as a flow node, wire it as:
            [Agent Loop] → [Reflection] → [Conditional Router (check_field: "satisfied")]
                ↑                                    |                    |
                |_____________ false ________________|          true      ↓
                (improvement msg already in messages)              [Chat Output]

        Input:
            - messages:  conversation history (from agent_loop output)
            - content:   assistant response text (set by agent_loop)

        Config:
            - default_reflection_prompt (str): system prompt for the reflection LLM call
            - inject_improvement (bool, default True):
                When not satisfied, append an improvement feedback message to
                'messages' so the agent_loop can act on it when retried.

        Output:
            - reflection (dict):  {satisfied, reason, needs_improvement}
            - satisfied (bool):   top-level flag for ConditionalRouterExecutor routing
            - messages (list):    unchanged if satisfied; improvement message appended if not
        """
        if input_data is None:
            input_data = {}

        config = config or {}

        user_request = self._extract_user_message(input_data)
        assistant_response = self._extract_assistant_response(input_data)

        if not user_request or not assistant_response:
            # Not enough data to evaluate — treat as satisfied to avoid blocking the flow
            result = input_data.copy()
            result["reflection"] = {
                "satisfied": True,
                "reason": "No content to evaluate",
                "needs_improvement": None
            }
            result["satisfied"] = True
            return result

        # Build reflection prompt
        reflection_prompt = config.get(
            "default_reflection_prompt",
            "You are a reflection agent. Evaluate if the assistant's response satisfies the user's request.\n\n"
            "Evaluate:\n"
            "- Is the request fully answered?\n"
            "- Are all tasks completed?\n"
            "- Is more work needed?\n\n"
            "Respond with JSON: "
            "{\"satisfied\": true/false, \"reason\": \"explanation\", "
            "\"needs_improvement\": \"what could be better or null\"}"
        )

        reflection_messages = [
            {"role": "system", "content": reflection_prompt},
            {
                "role": "user",
                "content": f"User Request: {user_request}\n\nAssistant Response: {assistant_response}"
            }
        ]

        try:
            response = await self.llm.chat_completion(
                messages=reflection_messages,
                model=settings.get("default_model"),
                temperature=0.1,
                max_tokens=300
            )

            reflection_result = {
                "satisfied": False,
                "reason": "Could not evaluate",
                "needs_improvement": None
            }

            if response and "choices" in response:
                raw = response["choices"][0]["message"].get("content", "")

                try:
                    # Strip JavaScript-style comments before parsing
                    cleaned = re.sub(r'//.*', '', raw)
                    cleaned = re.sub(r'/\*.*?\*/', '', cleaned, flags=re.DOTALL)
                    parsed = json.loads(cleaned.strip())

                    if isinstance(parsed, dict):
                        reflection_result = {
                            "satisfied": bool(parsed.get("satisfied", False)),
                            "reason": str(parsed.get("reason", "")),
                            "needs_improvement": parsed.get("needs_improvement")
                        }
                except (json.JSONDecodeError, IndexError):
                    pass

            result = input_data.copy()
            result["reflection"] = reflection_result
            result["satisfied"] = reflection_result["satisfied"]

            # When not satisfied and improvement hint is available, inject a
            # feedback message so the agent_loop can act on it on retry.
            inject = config.get("inject_improvement", True)
            needs_improvement = reflection_result.get("needs_improvement")
            if not reflection_result["satisfied"] and needs_improvement and inject:
                result = self._inject_improvement_message(result, needs_improvement)

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
