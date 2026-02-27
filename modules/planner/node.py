import json
import re
from core.llm import LLMBridge
from core.settings import settings


class PlannerExecutor:
    def __init__(self):
        self.llm = LLMBridge(
            base_url=settings.get("llm_api_url"),
            api_key=settings.get("llm_api_key")
        )

    def _extract_user_message(self, input_data: dict) -> str:
        """Extract the latest user message from input data."""
        if not input_data:
            return ""
        
        messages = input_data.get("messages", [])
        
        for msg in reversed(messages):
            if isinstance(msg, dict) and msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    return content
                elif isinstance(content, list):
                    text_parts = []
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            text_parts.append(part.get("text", ""))
                    return " ".join(text_parts)
        return ""

    async def receive(self, input_data: dict, config: dict = None) -> dict:
        """
        Creates a step-by-step plan for the user's request.
        
        Input:
            - messages: conversation history
            
        Output:
            - plan: array of steps with {step, action, target, depends_on}
            - current_step: 0
            - original_request: the user's request
            - plan_needed: true/false
            - plan_context: formatted plan string for system prompt
        """
        if input_data is None:
            input_data = {}
        
        config = config or {}
        
        if not config.get("enabled", True):
            return input_data
        
        user_request = self._extract_user_message(input_data)
        
        if not user_request:
            return input_data
        
        # Get planning prompt
        planner_prompt = config.get("default_planner_prompt", 
            "Create a plan for: {request}\nRespond with JSON array only.")
        
        max_steps = int(config.get("max_steps", 10))
        planner_prompt = planner_prompt.replace("{request}", user_request)
        
        planning_messages = [
            {"role": "system", "content": planner_prompt},
            {"role": "user", "content": user_request}
        ]
        
        plan = []
        
        try:
            response = await self.llm.chat_completion(
                messages=planning_messages,
                model=settings.get("default_model"),
                temperature=0.1,
                max_tokens=500
            )
            
            if response and "choices" in response:
                content = response["choices"][0]["message"].get("content", "")
                
                try:
                    # Remove JavaScript-style comments
                    content = re.sub(r'//.*', '', content)
                    content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
                    content = content.strip()
                    
                    parsed = json.loads(content)
                    
                    if isinstance(parsed, list):
                        plan = []
                        for i, step in enumerate(parsed[:max_steps]):
                            if isinstance(step, dict):
                                plan.append({
                                    "step": step.get("step", i + 1),
                                    "action": step.get("action", step.get("task", "unknown")),
                                    "target": step.get("target", step.get("query", step.get("details", ""))),
                                    "depends_on": step.get("depends_on")
                                })
                    elif isinstance(parsed, dict):
                        steps = parsed.get("plan", parsed.get("steps", []))
                        if isinstance(steps, list):
                            plan = []
                            for i, step in enumerate(steps[:max_steps]):
                                if isinstance(step, dict):
                                    plan.append({
                                        "step": step.get("step", i + 1),
                                        "action": step.get("action", step.get("task", "unknown")),
                                        "target": step.get("target", step.get("query", step.get("details", ""))),
                                        "depends_on": step.get("depends_on")
                                    })
                except (json.JSONDecodeError, IndexError):
                    pass
            
            result = input_data.copy()
            result["plan"] = plan
            result["current_step"] = 0
            result["original_request"] = user_request
            result["plan_needed"] = len(plan) > 0
            
            if plan:
                plan_text = "\n".join([
                    f"{p['step']}. {p['action']}: {p.get('target', '')}"
                    for p in plan
                ])
                result["plan_context"] = f"## Execution Plan\n{plan_text}"
            
            return result
            
        except Exception as e:
            result = input_data.copy()
            result["plan"] = []
            result["current_step"] = 0
            result["original_request"] = user_request
            result["plan_needed"] = False
            result["planning_error"] = str(e)
            return result

    async def send(self, processed_data: dict) -> dict:
        return processed_data


async def get_executor_class(node_type_id: str):
    if node_type_id == "planner":
        return PlannerExecutor
    return None
