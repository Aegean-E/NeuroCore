import json
import os
import asyncio
import httpx
from core.llm import LLMBridge
from core.settings import settings


TOOLS_FILE = "modules/tools/tools.json"
LIBRARY_DIR = os.path.join(os.path.dirname(__file__), "..", "tools", "library")


class AgentLoopExecutor:
    def __init__(self):
        self.llm = LLMBridge(
            base_url=settings.get("llm_api_url"),
            api_key=settings.get("llm_api_key")
        )

    def _load_tools(self):
        """Load tool definitions from tools.json."""
        if os.path.exists(TOOLS_FILE):
            with open(TOOLS_FILE, "r") as f:
                try:
                    return json.load(f)
                except:
                    return {}
        return {}

    def _load_tool_library(self):
        """Load tool implementations from library directory."""
        library = {}
        if os.path.exists(LIBRARY_DIR):
            for filename in os.listdir(LIBRARY_DIR):
                if filename.endswith(".py"):
                    tool_name = filename[:-3]
                    code_path = os.path.join(LIBRARY_DIR, filename)
                    with open(code_path, "r") as f:
                        library[tool_name] = f.read()
        return library

    async def _execute_tool(self, tool_call: dict, tool_library: dict):
        """Execute a single tool call."""
        func_name = tool_call["function"]["name"]
        args = json.loads(tool_call["function"]["arguments"])
        
        if func_name in tool_library:
            code = tool_library[func_name]
            local_scope = {"args": args, "result": None, "json": json, "httpx": httpx, "asyncio": asyncio}
            try:
                exec(code, local_scope)
                output = local_scope.get("result", "Success (no result returned)")
            except Exception as e:
                output = f"Error executing tool {func_name}: {str(e)}"
        else:
            output = f"Error: Tool {func_name} not found in library."
        
        return {
            "tool_call_id": tool_call["id"],
            "role": "tool",
            "name": func_name,
            "content": str(output)
        }

    def _build_system_prompt(self, input_data: dict, config: dict) -> str:
        """Build system prompt with context from previous nodes."""
        prompt_parts = []
        
        # Include plan context if available
        if config.get("include_plan_in_context", True):
            plan_context = input_data.get("plan_context")
            if plan_context:
                prompt_parts.append(plan_context)
        
        # Include memory context if available
        if config.get("include_memory_context", True):
            memory_context = input_data.get("_memory_context")
            if memory_context:
                prompt_parts.append(f"## User Memories\n{memory_context}")
        
        # Include knowledge context if available
        if config.get("include_knowledge_context", True):
            knowledge_context = input_data.get("knowledge_context")
            if knowledge_context:
                prompt_parts.append(f"## Relevant Knowledge\n{knowledge_context}")
        
        # Include reasoning context if available
        if config.get("include_reasoning_context", True):
            reasoning_context = input_data.get("reasoning_context")
            if reasoning_context:
                prompt_parts.append(f"## Previous Reasoning\n{reasoning_context}")
        
        if prompt_parts:
            return "\n\n".join(prompt_parts)
        return ""

    async def receive(self, input_data: dict, config: dict = None) -> dict:
        """
        Agent loop: repeatedly calls LLM with tools until no more tool calls.
        
        Input:
            - messages: conversation history (should include user message)
            - Optional: plan_context, _memory_context, knowledge_context, reasoning_context
            
        Output:
            - messages: full conversation with tool results
            - response: final LLM response
            - iterations: number of LLM ↔ Tool loops
        """
        if input_data is None:
            input_data = {}
        
        config = config or {}
        
        # Get configuration
        max_iterations = int(config.get("max_iterations", 10))
        max_tokens = int(config.get("max_tokens", 2048))
        temperature = float(config.get("temperature", 0.7))
        
        # Get messages
        messages = input_data.get("messages", [])
        if not messages:
            return input_data
        
        # Build system prompt with context
        system_prompt = self._build_system_prompt(input_data, config)
        
        # Load tools
        tools_def = self._load_tools()
        
        # Filter enabled tools only
        tools_list = []
        if tools_def:
            for tool_name, tool_data in tools_def.items():
                if isinstance(tool_data, dict) and tool_data.get("enabled", True):
                    definition = tool_data.get("definition")
                    if definition:
                        tools_list.append(definition)
        
        tool_library = self._load_tool_library()
        
        # Build messages for LLM
        llm_messages = messages.copy()
        
        # Add system prompt if we have context
        if system_prompt:
            has_system = any(m.get("role") == "system" for m in llm_messages)
            if has_system:
                for m in llm_messages:
                    if m.get("role") == "system":
                        m["content"] = m["content"] + "\n\n" + system_prompt
            else:
                llm_messages.insert(0, {"role": "system", "content": system_prompt})
        
        # Get model
        model = config.get("model") or settings.get("default_model")
        
        # Agent loop
        iterations = 0
        final_response = None
        
        for iteration in range(max_iterations):
            iterations += 1
            
            response = await self.llm.chat_completion(
                messages=llm_messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=tools_list if tools_list else None
            )
            
            if not response or "choices" not in response:
                break
            
            final_response = response
            assistant_message = response["choices"][0]["message"]
            
            # Add assistant message to conversation
            llm_messages.append(assistant_message)
            
            # Check for tool calls
            tool_calls = assistant_message.get("tool_calls", [])
            if not tool_calls:
                break
            
            # Execute tool calls and add results
            for tool_call in tool_calls:
                tool_result = await self._execute_tool(tool_call, tool_library)
                llm_messages.append(tool_result)
        
        # Build result
        result = input_data.copy()
        result["messages"] = llm_messages
        result["response"] = final_response
        result["iterations"] = iterations
        
        # Extract content for downstream nodes
        if final_response and "choices" in final_response:
            content = final_response["choices"][0]["message"].get("content", "")
            result["content"] = content
        
        return result

    async def send(self, processed_data: dict) -> dict:
        return processed_data


async def get_executor_class(node_type_id: str):
    if node_type_id == "agent_loop":
        return AgentLoopExecutor
    return None
