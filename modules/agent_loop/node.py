import json
import os
import asyncio
import logging
from core.llm import LLMBridge
from core.settings import settings
from modules.tools.sandbox import ToolSandbox


logger = logging.getLogger(__name__)

TOOLS_FILE = "modules/tools/tools.json"
LIBRARY_DIR = os.path.join(os.path.dirname(__file__), "..", "tools", "library")


class AgentLoopExecutor:
    # Class-level cache: avoids re-reading tools.json and library on every receive() call
    _tools_cache = {"mtime": 0.0, "data": {}}
    _library_cache = {"mtime": 0.0, "data": {}}

    def __init__(self):
        self.llm = LLMBridge(
            base_url=settings.get("llm_api_url"),
            api_key=settings.get("llm_api_key")
        )
        # Create sandbox instance for secure tool execution
        self._sandbox = ToolSandbox(timeout=30.0)

    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for a given text.
        Uses a rough approximation: ~4 characters per token.
        """
        if not text:
            return 0
        return len(text) // 4

    def _truncate_messages(
        self,
        messages: list,
        max_context_tokens: int,
        preserve_system: bool = True
    ) -> list:
        """
        Truncate messages to fit within token budget.
        Preserves system message and last N turns.
        """
        if not messages:
            return messages
            
        # Calculate current token count
        total_tokens = 0
        for msg in messages:
            content = msg.get("content", "")
            total_tokens += self._estimate_tokens(content)
        
        # If under limit, no truncation needed
        if total_tokens <= max_context_tokens:
            return messages
        
        # Find system message if present
        system_msg = None
        non_system_messages = []
        for msg in messages:
            if msg.get("role") == "system" and preserve_system:
                system_msg = msg
            else:
                non_system_messages.append(msg)
        
        # Keep last N non-system messages that fit in budget
        system_tokens = self._estimate_tokens(system_msg.get("content", "")) if system_msg else 0
        available_tokens = max_context_tokens - system_tokens
        
        result = []
        if system_msg:
            result.append(system_msg)
        
        # Add from end until we hit limit
        tokens_used = 0
        for msg in reversed(non_system_messages):
            msg_tokens = self._estimate_tokens(msg.get("content", ""))
            if tokens_used + msg_tokens <= available_tokens:
                result.insert(0 if system_msg else len(result), msg)
                tokens_used += msg_tokens
            else:
                break
        
        return result

    def _load_tools(self):
        """Load tool definitions from tools.json with mtime-based caching."""
        try:
            if os.path.exists(TOOLS_FILE):
                mtime = os.path.getmtime(TOOLS_FILE)
                if mtime > self.__class__._tools_cache["mtime"]:
                    with open(TOOLS_FILE, "r") as f:
                        self.__class__._tools_cache["data"] = json.load(f)
                    self.__class__._tools_cache["mtime"] = mtime
                return self.__class__._tools_cache["data"]
        except (json.JSONDecodeError, OSError, KeyError) as e:
            logger.warning(f"Failed to load tools: {e}")
        return {}

    def _load_tool_library(self):
        """Load tool implementations from library directory with mtime-based caching."""
        try:
            if os.path.exists(LIBRARY_DIR):
                # Check directory mtime
                dir_mtime = os.path.getmtime(LIBRARY_DIR)
                if dir_mtime > self.__class__._library_cache["mtime"]:
                    library = {}
                    for filename in os.listdir(LIBRARY_DIR):
                        if filename.endswith(".py"):
                            tool_name = filename[:-3]
                            code_path = os.path.join(LIBRARY_DIR, filename)
                            with open(code_path, "r") as f:
                                library[tool_name] = f.read()
                    self.__class__._library_cache["data"] = library
                    self.__class__._library_cache["mtime"] = dir_mtime
                return self.__class__._library_cache["data"]
        except OSError as e:
            logger.warning(f"Failed to load tool library: {e}")
        return {}

    async def _execute_tool(self, tool_call: dict, tool_library: dict) -> dict:
        """Execute a single tool call in sandbox and return the tool result message."""
        func_name = tool_call["function"]["name"]
        try:
            args = json.loads(tool_call["function"]["arguments"])
        except (json.JSONDecodeError, KeyError):
            args = {}

        success = False
        if func_name in tool_library:
            code = tool_library[func_name]
            try:
                # Run in executor to avoid blocking the event loop
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(
                    None,
                    self._sandbox.execute,
                    code,
                    {"args": args}
                )
                output = result.get("result", "Success (no result returned)")
                success = True
            except Exception as e:
                output = f"Error executing tool {func_name}: {str(e)}"
        else:
            output = f"Error: Tool {func_name} not found in library."

        return {
            "tool_call_id": tool_call.get("id", ""),
            "role": "tool",
            "name": func_name,
            "content": str(output),
            "success": success
        }

    def _build_system_prompt(self, input_data: dict, config: dict) -> str:
        """Build system prompt with context from previous nodes."""
        prompt_parts = []

        if config.get("include_plan_in_context", True):
            plan_context = input_data.get("plan_context")
            if plan_context:
                prompt_parts.append(plan_context)

        if config.get("include_memory_context", True):
            memory_context = input_data.get("_memory_context")
            if memory_context:
                prompt_parts.append(f"## User Memories\n{memory_context}")

        if config.get("include_knowledge_context", True):
            knowledge_context = input_data.get("knowledge_context")
            if knowledge_context:
                prompt_parts.append(f"## Relevant Knowledge\n{knowledge_context}")

        if config.get("include_reasoning_context", True):
            reasoning_context = input_data.get("reasoning_context")
            if reasoning_context:
                prompt_parts.append(f"## Previous Reasoning\n{reasoning_context}")

        return "\n\n".join(prompt_parts) if prompt_parts else ""

    async def _llm_with_retry(
        self,
        messages: list,
        model: str,
        temperature: float,
        max_tokens: int,
        tools: list,
        max_retries: int,
        retry_delay: float
    ) -> dict:
        """
        Call LLM with exponential backoff retry on failure.

        Retries when:
        - Response is None or empty
        - Response contains an 'error' key
        - Response has no 'choices' key

        Backoff formula: retry_delay * 2^(attempt - 1)

        Returns the response dict, or an error dict after all retries exhausted.
        """
        last_error = "Unknown error"

        for attempt in range(max_retries + 1):
            if attempt > 0:
                # Exponential backoff: delay doubles with each retry
                delay = retry_delay * (2 ** (attempt - 1))
                await asyncio.sleep(delay)

            try:
                response = await self.llm.chat_completion(
                    messages=messages,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    tools=tools if tools else None
                )

                # Valid response: has choices and no error
                if response and "choices" in response and not response.get("error"):
                    return response

                # Extract error for logging
                if response:
                    last_error = response.get("error", "Response missing 'choices' key")
                else:
                    last_error = "LLM returned None"

            except Exception as e:
                last_error = str(e)

        return {
            "error": f"LLM failed after {max_retries + 1} attempt(s): {last_error}"
        }

    async def _run_agent_loop(
        self,
        llm_messages: list,
        tools_list: list,
        tool_library: dict,
        model: str,
        temperature: float,
        max_tokens: int,
        max_iterations: int,
        max_llm_retries: int,
        retry_delay: float,
        tool_error_strategy: str,
        trace: list
    ) -> tuple:
        """
        Core agent loop: LLM ↔ Tool execution until no more tool calls or max_iterations.

        Args:
            llm_messages:       Mutable message list (modified in-place with assistant + tool messages)
            tools_list:         List of tool definitions to pass to LLM
            tool_library:       Dict of tool_name -> source code
            model:              LLM model name
            temperature:        Sampling temperature
            max_tokens:         Max tokens per LLM call
            max_iterations:     Maximum number of LLM ↔ Tool loops
            max_llm_retries:    Max retries per LLM call on failure
            retry_delay:        Base delay (seconds) for exponential backoff
            tool_error_strategy: "continue" (skip failed tools) or "stop" (abort on error)
            trace:              List to append per-iteration trace dicts to

        Returns:
            (final_response, iterations, had_tool_error)
        """
        iterations = 0
        final_response = None
        had_tool_error = False
        seen_calls = set()  # Track tool call signatures to detect loops

        for iteration in range(max_iterations):
            iterations += 1
            iteration_trace = {
                "iteration": iterations,
                "tool_calls": [],
                "errors": []
            }

            # Call LLM with retry logic
            response = await self._llm_with_retry(
                messages=llm_messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=tools_list,
                max_retries=max_llm_retries,
                retry_delay=retry_delay
            )

            # Check for LLM error after all retries exhausted
            if not response or "choices" not in response:
                error_msg = (
                    response.get("error", "LLM returned no choices")
                    if response else "LLM returned None"
                )
                iteration_trace["errors"].append(error_msg)
                trace.append(iteration_trace)
                break

            final_response = response
            assistant_message = response["choices"][0]["message"]

            # Append assistant message to conversation
            llm_messages.append(assistant_message)

            # Check for tool calls
            tool_calls = assistant_message.get("tool_calls", [])
            if not tool_calls:
                # No more tool calls — agent has finished
                trace.append(iteration_trace)
                break

            # Execute all tool calls in this turn
            tool_error_occurred = False
            for tool_call in tool_calls:
                tool_name = tool_call.get("function", {}).get("name", "unknown")
                tool_args = tool_call.get("function", {}).get("arguments", "{}")
                
                # Tool call deduplication: detect loops
                call_signature = f"{tool_name}:{tool_args}"
                if call_signature in seen_calls:
                    # Model is spinning - same tool call detected
                    iteration_trace["errors"].append(
                        f"Loop detected: '{tool_name}' called with same arguments repeatedly"
                    )
                    # Don't execute again, break the loop
                    break
                seen_calls.add(call_signature)
                
                iteration_trace["tool_calls"].append(tool_name)

                tool_result = await self._execute_tool(tool_call, tool_library)

                # Detect tool execution errors using structured success field
                if not tool_result.get("success", False):
                    had_tool_error = True
                    tool_error_occurred = True
                    iteration_trace["errors"].append(
                        f"Tool '{tool_name}': {tool_result['content']}"
                    )

                llm_messages.append(tool_result)

            trace.append(iteration_trace)

            # Apply tool error strategy
            if tool_error_occurred and tool_error_strategy == "stop":
                break

        return final_response, iterations, had_tool_error

    async def receive(self, input_data: dict, config: dict = None) -> dict:
        """
        Agent loop: repeatedly calls LLM with tools until no more tool calls.

        Input:
            - messages: conversation history (must include at least one user message)
            - Optional context keys: plan_context, _memory_context,
              knowledge_context, reasoning_context

        Config:
            - max_iterations (int, default 10):
                Maximum number of LLM ↔ Tool loops per run.
            - max_tokens (int, default 2048):
                Max tokens per LLM call.
            - temperature (float, default 0.7):
                Sampling temperature.
            - max_llm_retries (int, default 3):
                Number of retries on LLM failure (exponential backoff).
            - retry_delay (float, default 1.0):
                Base delay in seconds for exponential backoff between retries.
            - tool_error_strategy (str, default "continue"):
                "continue" — skip failed tools and keep looping.
                "stop"     — abort the loop on the first tool error.
            - timeout (float, default 120):
                Total timeout in seconds for the entire agent loop (0 = disabled).
            - max_replan_depth (int, default 3):
                Maximum number of re-planning cycles allowed before hard-stopping.
            - include_plan_in_context (bool, default True)
            - include_memory_context (bool, default True)
            - include_knowledge_context (bool, default True)
            - include_reasoning_context (bool, default True)

        Output:
            - messages (list):          Full conversation including tool results.
            - response (dict):          Raw final LLM response.
            - content (str):            Extracted text content from final response.
            - iterations (int):         Total number of LLM ↔ Tool loops executed.
            - agent_loop_trace (list):  Per-iteration details (tool_calls, errors).
            - agent_loop_error (str):   Error message if the loop failed or timed out.
            - replan_needed (bool):     True if execution failed and re-planning is recommended.
            - replan_reason (str):      Explanation of why re-planning is needed.
            - suggested_approach (str): Suggestion for how to re-plan.
            - replan_count (int):        Current re-planning depth counter (increments each re-plan).

        Reflection-driven retry is handled externally by wiring:
            [Agent Loop] → [Reflection] → [Conditional Router (satisfied)] → [Agent Loop]
        
        Re-planning depth protection:
            - Tracks replan_count across iterations
            - Hard-stops when replan_count >= max_replan_depth
            - Surfaces degraded result with error rather than infinite looping
        """
        if input_data is None:
            input_data = {}

        config = config or {}

        # --- Read configuration ---
        max_iterations = int(config.get("max_iterations", 10))
        max_tokens = int(config.get("max_tokens", 2048))
        temperature = float(config.get("temperature", 0.7))
        max_llm_retries = int(config.get("max_llm_retries", 3))
        retry_delay = float(config.get("retry_delay", 1.0))
        tool_error_strategy = str(config.get("tool_error_strategy", "continue"))
        timeout = float(config.get("timeout", 120))
        max_replan_depth = int(config.get("max_replan_depth", 3))
        max_context_tokens = int(config.get("max_context_tokens", 6000))  # Token budget for context
        
        # --- Re-planning depth tracking ---
        replan_count = int(input_data.get("replan_count", 0))
        
        # Check if we've exceeded max re-planning depth (only if max_replan_depth > 0)
        if max_replan_depth > 0 and replan_count >= max_replan_depth:
            result = input_data.copy()
            result["replan_needed"] = False
            result["replan_depth_exceeded"] = True
            result["agent_loop_error"] = f"Max re-planning depth ({max_replan_depth}) exceeded. Unable to complete task."
            result["content"] = f"[Error: Task failed after {replan_count} re-planning attempts. Unable to find viable execution path.]"
            result["iterations"] = 0
            result["agent_loop_trace"] = []
            return result

        # Guard: no messages → return input unchanged
        messages = input_data.get("messages", [])
        if not messages:
            return input_data

        # Build system prompt from context fields
        system_prompt = self._build_system_prompt(input_data, config)

        # Load tools
        tools_def = self._load_tools()
        tools_list = []
        if tools_def:
            for tool_name, tool_data in tools_def.items():
                if isinstance(tool_data, dict) and tool_data.get("enabled", True):
                    definition = tool_data.get("definition")
                    if definition:
                        tools_list.append(definition)

        tool_library = self._load_tool_library()
        model = config.get("model") or settings.get("default_model")

        # Shared trace list (populated by _run_agent_loop)
        agent_loop_trace = []

        async def _execute():
            """Inner coroutine — wrapped with optional timeout."""
            # Build initial message list
            llm_messages = messages.copy()

            # Apply token budget truncation to prevent context overflow
            if max_context_tokens > 0:
                llm_messages = self._truncate_messages(llm_messages, max_context_tokens)

            # Inject system prompt if context is available
            # Skip if context is already present (avoid duplication from SystemPromptExecutor)
            if system_prompt:
                has_system = any(m.get("role") == "system" for m in llm_messages)
                if has_system:
                    # Check if any of the context keys are already in the system message
                    system_msg = next((m for m in llm_messages if m.get("role") == "system"), {})
                    existing_content = system_msg.get("content", "")
                    
                    # Check if context markers already exist to avoid duplication
                    context_markers = ["## Execution Plan", "## User Memories", "## Relevant Knowledge", "## Previous Reasoning"]
                    context_already_present = any(marker in existing_content for marker in context_markers)
                    
                    if not context_already_present:
                        for m in llm_messages:
                            if m.get("role") == "system":
                                m["content"] = m["content"] + "\n\n" + system_prompt
                                break
                else:
                    llm_messages.insert(0, {"role": "system", "content": system_prompt})

            final_response, iterations, had_tool_error = await self._run_agent_loop(
                llm_messages=llm_messages,
                tools_list=tools_list,
                tool_library=tool_library,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                max_iterations=max_iterations,
                max_llm_retries=max_llm_retries,
                retry_delay=retry_delay,
                tool_error_strategy=tool_error_strategy,
                trace=agent_loop_trace
            )

            result = input_data.copy()
            result["messages"] = llm_messages
            result["response"] = final_response
            result["iterations"] = iterations

            if final_response and "choices" in final_response:
                result["content"] = final_response["choices"][0]["message"].get("content", "")

            # --- Re-planning detection ---
            # Check if the agent actually failed to produce a useful response
            # Don't trigger re-planning just because max_iterations was reached
            content = result.get("content", "")
            response_text = str(final_response) if final_response else ""
            actually_failed = had_tool_error or not content or "error" in response_text.lower()
            
            plan = input_data.get("plan", [])
            current_step = input_data.get("current_step", 0)
            
            # Check if re-planning is needed
            if actually_failed:
                result["replan_needed"] = True
                # Increment replan count for depth tracking
                result["replan_count"] = replan_count + 1
                
                if had_tool_error:
                    result["replan_reason"] = "Tool execution errors occurred during plan execution"
                elif not content:
                    result["replan_reason"] = "Agent produced no content response"
                else:
                    result["replan_reason"] = "Agent response contained error indicators"
                
                # Suggest simpler approach if plan has multiple steps
                if plan and len(plan) > 1:
                    result["suggested_approach"] = (
                        f"Current plan has {len(plan)} steps. "
                        f"Consider breaking into smaller sub-tasks or simplifying step {current_step + 1}."
                    )
                elif plan:
                    result["suggested_approach"] = (
                        "Single-step plan failed. Consider using different tools or approach."
                    )
                else:
                    result["suggested_approach"] = "No plan exists. Consider creating a step-by-step plan."
            else:
                result["replan_needed"] = False
                # Reset replan count on successful execution
                result["replan_count"] = 0

            return result

        # --- Execute with optional timeout ---
        try:
            if timeout > 0:
                result = await asyncio.wait_for(_execute(), timeout=timeout)
            else:
                result = await _execute()

        except asyncio.TimeoutError:
            result = input_data.copy()
            result["agent_loop_error"] = f"Agent loop timed out after {timeout}s"
            result["iterations"] = 0
            result["agent_loop_trace"] = agent_loop_trace
            
            # Re-planning needed due to timeout
            plan = input_data.get("plan", [])
            result["replan_needed"] = True
            result["replan_count"] = replan_count + 1
            result["replan_reason"] = f"Execution timed out after {timeout} seconds"
            if plan and len(plan) > 1:
                result["suggested_approach"] = (
                    f"Plan with {len(plan)} steps took too long. "
                    "Consider breaking into smaller chunks with shorter timeout."
                )
            else:
                result["suggested_approach"] = "Execution timed out. Consider simplifying the approach."
            
            return result

        except Exception as e:
            result = input_data.copy()
            result["agent_loop_error"] = str(e)
            result["iterations"] = 0
            result["agent_loop_trace"] = agent_loop_trace
            
            # Re-planning needed due to error
            plan = input_data.get("plan", [])
            result["replan_needed"] = True
            result["replan_count"] = replan_count + 1
            result["replan_reason"] = f"Unexpected error: {str(e)}"
            result["suggested_approach"] = "Review plan and try alternative approach."
            
            return result

        # Attach trace to final result
        result["agent_loop_trace"] = agent_loop_trace
        return result

    async def send(self, processed_data: dict) -> dict:
        return processed_data


async def get_executor_class(node_type_id: str):
    if node_type_id == "agent_loop":
        return AgentLoopExecutor
    return None

