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
        # Lazy import to avoid circular dependencies
        self._reasoning_service = None
    
    def _get_reasoning_service(self):
        """Lazy load ReasoningBookService to avoid circular imports."""
        if self._reasoning_service is None:
            try:
                from modules.reasoning_book.service import service as reasoning_service
                self._reasoning_service = reasoning_service
            except ImportError:
                self._reasoning_service = None
        return self._reasoning_service

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
        
        # Get planning prompt - check for custom prompt first, then default
        planner_prompt = config.get("planner_prompt")  # Custom user-defined prompt
        if not planner_prompt:
            planner_prompt = config.get(
                "default_planner_prompt",
                "You are a task planner. Analyze the user's request and create a step-by-step execution plan if needed.\n\n"
                "Rules:\n"
                "- If the request is simple (one question, greeting, or single action), return an empty JSON array: []\n"
                "- If the request requires multiple steps, return a JSON array with each step: [{\"action\": \"step description\", \"target\": \"what it applies to\"}]\n"
                "- Keep plans simple and actionable\n"
                "- Maximum {max_steps} steps\n\n"
                "Request: {request}\n\n"
                "Respond with JSON array only."
            )
        
        # Include reasoning context if enabled and available
        if config.get("include_reasoning_context", False):
            reasoning_context = input_data.get("reasoning_context", "")
            if reasoning_context:
                planner_prompt += f"\n\nPrevious reasoning context:\n{reasoning_context}\n\nConsider this context when creating your plan."
        
        max_steps = int(config.get("max_steps", 10))
        planner_prompt = planner_prompt.replace("{request}", user_request)
        planner_prompt = planner_prompt.replace("{max_steps}", str(max_steps))
        
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
            
            # Log plan to Reasoning Book if enabled
            if config.get("log_to_reasoning_book", False):
                reasoning_service = self._get_reasoning_service()
                if reasoning_service:
                    plan_summary = f"Created {len(plan)}-step plan for: {user_request[:100]}..."
                    if plan:
                        plan_summary += f"\nSteps: {', '.join([p['action'] for p in plan[:3]])}"
                        if len(plan) > 3:
                            plan_summary += f" and {len(plan) - 3} more"
                    reasoning_service.log_thought(plan_summary, source="Planner")
            
            return result
            
        except Exception as e:
            result = input_data.copy()
            result["plan"] = []
            result["current_step"] = 0
            result["original_request"] = user_request
            result["plan_needed"] = False
            result["planning_error"] = str(e)
            
            # Log error to Reasoning Book if enabled
            if config.get("log_to_reasoning_book", False):
                reasoning_service = self._get_reasoning_service()
                if reasoning_service:
                    reasoning_service.log_thought(f"Planning failed for: {user_request[:100]}... Error: {str(e)}", source="Planner")
            
            return result

    async def send(self, processed_data: dict) -> dict:
        return processed_data


async def get_executor_class(node_type_id: str):
    if node_type_id == "planner":
        return PlannerExecutor
    return None
