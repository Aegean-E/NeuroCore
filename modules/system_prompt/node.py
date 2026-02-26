import json
import os

class SystemPromptExecutor:
    def _load_available_tools(self):
        """Load available tools from the tools module."""
        tools_file = os.path.join(os.path.dirname(__file__), "..", "tools", "tools.json")
        if os.path.exists(tools_file):
            try:
                with open(tools_file, "r") as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def _format_tools_section(self, enabled_tool_names: list) -> str:
        """Format available tools into a readable section for the system prompt."""
        all_tools = self._load_available_tools()
        
        if not enabled_tool_names:
            return ""
        
        tools_text = "\n## Available Tools\nYou have access to the following tools:\n"
        
        for tool_name in enabled_tool_names:
            if tool_name in all_tools:
                tool_data = all_tools[tool_name]
                if "definition" in tool_data and "function" in tool_data["definition"]:
                    func_def = tool_data["definition"]["function"]
                    desc = func_def.get("description", "No description available")
                    tools_text += f"- **{tool_name}**: {desc}\n"
        
        return tools_text
    
    def _get_tools_in_openai_format(self, enabled_tool_names: list) -> list:
        """Convert enabled tools to OpenAI tool format."""
        all_tools = self._load_available_tools()
        tools_list = []
        
        for tool_name in enabled_tool_names:
            if tool_name in all_tools and "definition" in all_tools[tool_name]:
                tools_list.append(all_tools[tool_name]["definition"])
        
        return tools_list

    async def receive(self, input_data: dict, config: dict = None) -> dict:
        """
        Receives the current conversation state and prepends a system message with available tools.
        Also passes tools in OpenAI format for the LLM to use.
        """
        if input_data is None:
            input_data = {}
            
        config = config or {}
        # Default prompt if none is configured
        prompt_text = config.get("system_prompt", "You are NeuroCore, a helpful and intelligent AI assistant.")
        
        # Get enabled tools from config
        enabled_tools = config.get("enabled_tools", [])
        tools_section = self._format_tools_section(enabled_tools)
        
        # Combine prompt with tools section
        full_prompt = prompt_text + tools_section
        
        # Check for injected context from bridge nodes (memory, knowledge, reasoning)
        # These are injected by Memory Recall, Query Knowledge, and Reasoning Load nodes
        
        # Memory context (injected as _memory_context by memory_recall)
        memory_context = input_data.get("_memory_context")
        if not memory_context:
            messages = input_data.get("messages", [])
            for msg in messages:
                if isinstance(msg, dict) and msg.get("role") == "system":
                    content = msg.get("content", "")
                    if "Relevant memories retrieved" in content:
                        memory_context = content
                        break
        
        # Knowledge context (from query_knowledge node)
        knowledge_context = input_data.get("knowledge_context")
        
        # Reasoning context (from reasoning_load node)
        reasoning_context = input_data.get("reasoning_context")
        
        # Build context sections
        context_parts = []
        if reasoning_context:
            context_parts.append(f"## Previous Reasoning\n{reasoning_context}")
        if knowledge_context:
            context_parts.append(f"## Relevant Knowledge\n{knowledge_context}")
        if memory_context:
            # Extract just the memory content (after "Relevant memories retrieved...")
            if "Relevant memories retrieved" in memory_context:
                context_parts.append(f"## User Memories\n{memory_context}")
            else:
                context_parts.append(f"## User Memories\n{memory_context}")
        
        if context_parts:
            full_prompt = full_prompt + "\n\n" + "\n\n".join(context_parts)
        
        # Get existing messages from the flow data
        messages = input_data.get("messages")
        if not isinstance(messages, list):
            messages = []
        
        # Create the system message
        system_message = {"role": "system", "content": full_prompt}
        
        # Prepend the system message to the history
        new_messages = [system_message] + messages
        
        # Get tools in OpenAI format for the LLM to use
        tools = self._get_tools_in_openai_format(enabled_tools)
        
        # Return the updated data structure with tools
        # We use **input_data to preserve any other keys flowing through the system
        result = {**input_data, "messages": new_messages}
        if tools:
            result["tools"] = tools
            result["available_tools"] = enabled_tools
        
        return result

    async def send(self, processed_data: dict) -> dict:
        return processed_data

async def get_executor_class(node_type_id: str):
    if node_type_id == "system_prompt":
        return SystemPromptExecutor
    return None