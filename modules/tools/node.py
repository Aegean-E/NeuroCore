import json
import os
import asyncio
import httpx
from .router import TOOLS_FILE, LIBRARY_DIR

class ToolDispatcherExecutor:
    def _load_tools(self):
        if os.path.exists(TOOLS_FILE):
            with open(TOOLS_FILE, "r") as f:
                try: return json.load(f)
                except: return {}
        return {}

    async def receive(self, input_data: dict, config: dict = None) -> dict:
        if not input_data or "choices" not in input_data:
            return input_data

        message = input_data["choices"][0]["message"]
        tool_calls = message.get("tool_calls")

        if not tool_calls:
            return input_data

        library = self._load_tools()
        allowed_tools = config.get("allowed_tools", []) # From node config
        max_tools_per_turn = config.get("max_tools_per_turn", 5) # Default 5 tools per turn
        results = []
        
        # Track tool count in context if available
        tool_count = input_data.get("_tool_count", 0)

        # Limit tool calls to max_tools_per_turn
        tool_calls_to_run = tool_calls[:max_tools_per_turn]
        remaining_tools = tool_calls[max_tools_per_turn:]
        
        # Store remaining tools for next turn
        if remaining_tools:
            input_data["_remaining_tool_calls"] = remaining_tools
        
        # Track tool count
        input_data["_tool_count"] = tool_count + len(tool_calls_to_run)
        
        # Check if we've hit the limit
        should_continue = len(tool_calls_to_run) < len(tool_calls)

        for tool_call in tool_calls_to_run:
            func_name = tool_call["function"]["name"]
            args = json.loads(tool_call["function"]["arguments"])
            
            # Check if tool is allowed by this specific dispatcher instance
            if allowed_tools and func_name not in allowed_tools:
                output = f"Error: Tool {func_name} is not enabled for this dispatcher."
            elif func_name in library:
                # Load code from the library directory
                code_path = os.path.join(os.path.dirname(__file__), "library", f"{func_name}.py")
                code = ""
                if os.path.exists(code_path):
                    with open(code_path, "r") as f:
                        code = f.read()
                # Execute custom tool logic
                local_scope = {"args": args, "result": None, "json": json, "httpx": httpx}
                try:
                    # We run this in a thread if it's heavy, but for now simple exec
                    exec(code, local_scope)
                    output = local_scope.get("result", "Success (no result returned)")
                except Exception as e:
                    output = f"Error executing tool {func_name}: {str(e)}"
            else:
                output = f"Error: Tool {func_name} not found in library."

            results.append({
                "tool_call_id": tool_call["id"],
                "role": "tool",
                "name": func_name,
                "content": str(output)
            })

        # Update the conversation history if provided
        messages = []
        if "messages" in input_data:
            messages = list(input_data["messages"])
            messages.append(message)
            messages.extend(results)

        return {
            "tool_results": results,
            "assistant_message": message,
            "messages": messages,
            "requires_continuation": not should_continue
        }

    async def send(self, processed_data: dict) -> dict:
        return processed_data

async def get_executor_class(node_type_id: str):
    if node_type_id == "tool_dispatcher":
        return ToolDispatcherExecutor
    return None