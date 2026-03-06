import asyncio
import json
import os
import logging

logger = logging.getLogger(__name__)

class SystemPromptExecutor:
    # Module-level cache with asyncio lock for thread-safety
    _tools_lock = asyncio.Lock()
    _tools_cache = {"mtime": 0.0, "data": {}}
    _tools_path = os.path.join(os.path.dirname(__file__), "..", "tools", "tools.json")

    async def _load_available_tools(self):
        """Load available tools from the tools module (cached by mtime with async lock)."""
        try:
            if os.path.exists(self._tools_path):
                mtime = os.path.getmtime(self._tools_path)
                async with self._tools_lock:
                    if mtime > self._tools_cache["mtime"]:
                        with open(self._tools_path, "r") as f:
                            self._tools_cache["data"] = json.load(f)
                        self._tools_cache["mtime"] = mtime
                return self._tools_cache["data"]
        except (json.JSONDecodeError, OSError, KeyError) as e:
            # JSONDecodeError: Corrupted JSON file
            # OSError: File read permissions or I/O issues
            # KeyError: Missing expected keys in JSON structure
            logger.warning(f"Failed to load available tools: {e}")
        return {}

    async def _get_tools_in_openai_format(self, enabled_tool_names: list) -> list:
        """Convert enabled tools to OpenAI tool format."""
        all_tools = await self._load_available_tools()
        tools_list = []
        
        for tool_name in enabled_tool_names:
            if tool_name in all_tools and "definition" in all_tools[tool_name]:
                tools_list.append(all_tools[tool_name]["definition"])
        
        return tools_list

    def _load_skills_content(self, enabled_skills: list) -> str:
        """Load content from enabled skills."""
        if not enabled_skills:
            return ""
        
        skills_content = []
        # Use absolute path based on this file's location for consistency
        storage_path = os.path.join(os.path.dirname(__file__), "..", "skills", "data")
        metadata_file = os.path.join(storage_path, "skills_metadata.json")
        
        # Load metadata to get skill names
        metadata = {}
        try:
            if os.path.exists(metadata_file):
                with open(metadata_file, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
        except (json.JSONDecodeError, OSError, KeyError) as e:
            # JSONDecodeError: Corrupted JSON file
            # OSError: File read permissions or I/O issues
            # KeyError: Missing expected keys in JSON structure
            logger.warning(f"Failed to load skills metadata: {e}")
        
        for skill_id in enabled_skills:
            skill_path = os.path.join(storage_path, f"{skill_id}.md")
            if os.path.exists(skill_path):
                try:
                    with open(skill_path, "r", encoding="utf-8") as f:
                        content = f.read()
                        skill_name = metadata.get(skill_id, {}).get("name", skill_id)
                        skills_content.append(f"### {skill_name}\n{content}")
                except (OSError, UnicodeDecodeError) as e:
                    # OSError: File read permissions or I/O issues
                    # UnicodeDecodeError: File encoding issues
                    logger.warning(f"Failed to load skill {skill_id}: {e}")
                    continue
        
        if skills_content:
            return "## Skills and Guidelines\n\n" + "\n\n".join(skills_content)
        return ""

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
        
        # Get enabled tools from config (structured format only - no markdown duplication)
        enabled_tools = config.get("enabled_tools", [])
        
        # Check for injected context from bridge nodes (memory, knowledge, reasoning)
        # These are injected by Memory Recall, Query Knowledge, and Reasoning Load nodes
        
        # Memory context (injected as _memory_context by memory_recall) - HIGH PRIORITY
        memory_context = input_data.get("_memory_context")
        
        # Knowledge context (from query_knowledge node) - MEDIUM PRIORITY
        knowledge_context = input_data.get("knowledge_context")
        
        # Reasoning context (from reasoning_load node) - MEDIUM PRIORITY
        reasoning_context = input_data.get("reasoning_context")
        
        # Plan context (from planner node) - HIGH PRIORITY
        plan_context = input_data.get("plan_context")
        
        # Skills context (from enabled_skills config) - LOW PRIORITY
        enabled_skills = config.get("enabled_skills", [])
        skills_context = self._load_skills_content(enabled_skills)
        
        # Build context sections with priority-based ordering
        # Priority levels: 0=highest (plan, memory), 1=medium (knowledge, reasoning), 2=lowest (skills)
        # Use tuples of (priority, content) for priority-aware budget management
        context_parts = []
        
        # High priority (0): Plan and Memory (user-specific)
        if plan_context:
            context_parts.append((0, plan_context))
        if memory_context:
            # Extract just the memory content (after "Relevant memories retrieved...")
            if "Relevant memories retrieved" in memory_context:
                # Split on the marker and take the content after it
                memory_content = memory_context.split("Relevant memories retrieved", 1)[1].strip()
                context_parts.append((0, f"## User Memories\n{memory_content}"))
            else:
                context_parts.append((0, f"## User Memories\n{memory_context}"))
        
        # Medium priority (1): Knowledge and Reasoning
        if knowledge_context:
            context_parts.append((1, f"## Relevant Knowledge\n{knowledge_context}"))
        if reasoning_context:
            context_parts.append((1, f"## Previous Reasoning\n{reasoning_context}"))
        
        # Low priority (2): Skills (generic, can be dropped if needed)
        if skills_context:
            context_parts.append((2, skills_context))
        
        # Apply token budget management
        max_token_budget = config.get("max_token_budget", 4000)
        full_prompt = self._apply_token_budget(prompt_text, context_parts, max_token_budget)
        
        # Get existing messages from the flow data
        messages = input_data.get("messages")
        if not isinstance(messages, list):
            messages = []
        
        # Create the system message
        system_message = {"role": "system", "content": full_prompt}
        
        # Prepend the system message to the history
        new_messages = [system_message] + messages
        
        # Get tools in OpenAI format for the LLM to use
        tools = await self._get_tools_in_openai_format(enabled_tools)
        
        # Return the updated data structure with tools
        # We use **input_data to preserve any other keys flowing through the system
        result = {**input_data, "messages": new_messages}
        if tools:
            result["tools"] = tools
            result["available_tools"] = enabled_tools
        
        return result

    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimation: ~4 characters per token."""
        return len(text) // 4

    def _apply_token_budget(self, base_prompt: str, context_parts: list, max_tokens: int) -> str:
        """
        Apply token budget management with priority-based trimming.
        Priority levels: 0=highest (plan, memory), 1=medium (knowledge, reasoning), 2=lowest (skills)
        
        Uses priority tags instead of indices to handle variable context parts.
        """
        base_tokens = self._estimate_tokens(base_prompt)
        available_tokens = max_tokens - base_tokens - 100  # Reserve 100 tokens for safety
        
        if available_tokens <= 0:
            # Base prompt alone exceeds budget, return truncated base
            return base_prompt[:max_tokens * 4]  # Rough char limit
        
        result_parts = []
        remaining_tokens = available_tokens
        
        # Sort by priority (lower number = higher priority)
        # This ensures we keep high-priority items when trimming
        sorted_parts = sorted(context_parts, key=lambda x: x[0])
        
        for priority, part in sorted_parts:
            part_tokens = self._estimate_tokens(part)
            
            # Check if this part fits within remaining budget
            if part_tokens <= remaining_tokens:
                result_parts.append(part)
                remaining_tokens -= part_tokens
            else:
                # Part doesn't fit - try to truncate if we have some remaining budget
                # Only truncate if this isn't the highest priority item (priority > 0)
                if remaining_tokens > 200 and priority > 0:
                    # Truncate with ellipsis
                    max_chars = remaining_tokens * 4 - 20  # Reserve for ellipsis
                    truncated = part[:max_chars] + "\n\n[... content truncated due to token budget]"
                    result_parts.append(truncated)
                    remaining_tokens = 0
                    break  # Short-circuit after truncation since we used up budget
                
                # For lowest priority parts (priority == 2, skills), skip entirely if they don't fit
                if priority >= 2:
                    logger.debug(f"Dropping lowest priority context (skills) due to token budget")
                    break
                # For medium priority (priority == 1), log but continue to lower priority
                elif priority == 1:
                    logger.debug(f"Dropping medium priority context (knowledge/reasoning) due to token budget")
        
        if result_parts:
            return base_prompt + "\n\n" + "\n\n".join(result_parts)
        return base_prompt

    async def send(self, processed_data: dict) -> dict:
        return processed_data

async def get_executor_class(node_type_id: str):
    if node_type_id == "system_prompt":
        return SystemPromptExecutor
    return None

