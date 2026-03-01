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
                except Exception:
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

    async def _execute_tool(self, tool_call: dict, tool_library: dict) -> dict:
        """Execute a single tool call and return the tool result message."""
        func_name = tool_call["function"]["name"]
        try:
            args = json.loads(tool_call["function"]["arguments"])
        except (json.JSONDecodeError, KeyError):
            args = {}

        if func_name in tool_library:
            code = tool_library[func_name]
            local_scope = {
                "args": args,
                "result": None,
                "json": json,
                "httpx": httpx,
                "asyncio": asyncio
            }
            try:
                exec(code, local_scope)
                output = local_scope.get("result", "Success (no result returned)")
            except Exception as e:
                output = f"Error executing tool {func_name}: {str(e)}"
        else:
            output = f"Error: Tool {func_name} not found in library."

        return {
            "tool_call_id": tool_call.get("id", ""),
            "role": "tool",
            "name": func_name,
            "content": str(output)
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
                iteration_trace["tool_calls"].append(tool_name)

                tool_result = await self._execute_tool(tool_call, tool_library)

                # Detect tool execution errors (tool outputs starting with "Error")
                if str(tool_result["content"]).startswith("Error"):
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

        Reflection-driven retry is handled externally by wiring:
            [Agent Loop] → [Reflection] → [Conditional Router (satisfied)] → [Agent Loop]
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

            # Inject system prompt if context is available
            if system_prompt:
                has_system = any(m.get("role") == "system" for m in llm_messages)
                if has_system:
                    for m in llm_messages:
                        if m.get("role") == "system":
                            m["content"] = m["content"] + "\n\n" + system_prompt
                            break
                else:
                    llm_messages.insert(0, {"role": "system", "content": system_prompt})

            final_response, iterations, _ = await self._run_agent_loop(
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
            return result

        except Exception as e:
            result = input_data.copy()
            result["agent_loop_error"] = str(e)
            result["iterations"] = 0
            result["agent_loop_trace"] = agent_loop_trace
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
