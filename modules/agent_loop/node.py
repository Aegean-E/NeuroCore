import json
import os
import asyncio
import logging
import time
from core.llm import LLMBridge
from core.settings import settings
from modules.tools.sandbox import ToolSandbox
from core.session_manager import session_manager, get_session_manager, EpisodeState


logger = logging.getLogger(__name__)

TOOLS_FILE = "modules/tools/tools.json"
LIBRARY_DIR = os.path.join(os.path.dirname(__file__), "..", "tools", "library")
RLM_LIBRARY_DIR = os.path.join(os.path.dirname(__file__), "..", "tools", "rlm_library")


# ---------------------------------------------------------------------------
# Shared base class
# ---------------------------------------------------------------------------

class AgentBaseExecutor:
    """Shared infrastructure for all agent loop variants.

    Provides: LLM bridge, sandbox, session manager, token estimation,
    message truncation, context compaction, tool schema loading, system
    prompt assembly, LLM retry wrapper, and async SubCall execution.
    """

    # Shared tool-schema cache.  All variants read the same tools.json so
    # sharing the cache eliminates redundant file reads.
    _tools_cache: dict = {"mtime": 0.0, "data": {}}

    def __init__(self):
        self.llm = LLMBridge(
            base_url=settings.get("llm_api_url"),
            api_key=settings.get("llm_api_key"),
        )
        self._sandbox = ToolSandbox(timeout=30.0)
        try:
            self._session_manager = get_session_manager()
        except Exception:
            self._session_manager = None

    # ------------------------------------------------------------------
    # Token helpers
    # ------------------------------------------------------------------

    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimate: ~4 characters per token."""
        if not text:
            return 0
        return len(text) // 4

    def _estimate_messages_tokens(self, messages: list) -> int:
        """Estimate total token count across all messages.

        Accounts for text content, tool_calls JSON on assistant messages,
        and tool result content (always a string, but may be JSON).
        """
        total = 0
        for m in messages:
            content = m.get("content") or ""
            if isinstance(content, list):
                content = " ".join(
                    p.get("text", "") for p in content if p.get("type") == "text"
                )
            total += self._estimate_tokens(str(content))
            tool_calls = m.get("tool_calls")
            if tool_calls:
                try:
                    total += self._estimate_tokens(json.dumps(tool_calls))
                except (TypeError, ValueError):
                    pass
        return total

    def _truncate_messages(
        self,
        messages: list,
        max_context_tokens: int,
        preserve_system: bool = True,
    ) -> list:
        """Truncate messages to fit within token budget.

        Preserves the system message and as many recent turns as fit.
        """
        if not messages:
            return messages

        total_tokens = sum(
            self._estimate_tokens(m.get("content", "")) for m in messages
        )
        if total_tokens <= max_context_tokens:
            return messages

        system_msg = None
        non_system = []
        for msg in messages:
            if msg.get("role") == "system" and preserve_system:
                system_msg = msg
            else:
                non_system.append(msg)

        system_tokens = (
            self._estimate_tokens(system_msg.get("content", "")) if system_msg else 0
        )
        available = max_context_tokens - system_tokens

        result = []
        used = 0
        for msg in reversed(non_system):
            t = self._estimate_tokens(msg.get("content", ""))
            if used + t <= available:
                result.append(msg)
                used += t
            else:
                break
        result.reverse()
        if system_msg:
            result.insert(0, system_msg)
        return result

    async def _compact_messages(self, messages: list, keep_last: int) -> list:
        """LLM-summarize old agent messages to free up context window.

        Preserves leading system messages verbatim.  Summarises the oldest
        non-system messages into a single system-level summary, then appends
        the most recent *keep_last* turns verbatim.  Returns the original list
        unchanged if compaction is unnecessary or the LLM call fails (fail-safe:
        never discard context silently).

        Summary budget scales with the number of messages being summarized so
        that long agent runs don't lose critical facts.
        """
        system_prefix = []
        conversation = []
        for msg in messages:
            if msg.get("role") == "system" and not conversation:
                system_prefix.append(msg)
            else:
                conversation.append(msg)

        if len(conversation) <= keep_last + 2:
            return messages

        old_messages = conversation[:-keep_last]
        recent_messages = conversation[-keep_last:]

        lines = []
        for m in old_messages:
            role = m.get("role", "?").upper()
            content = m.get("content") or ""
            if isinstance(content, list):
                content = " ".join(
                    p.get("text", "") for p in content if p.get("type") == "text"
                ) or "[non-text content]"
            content_str = str(content)[:600]
            tool_calls = m.get("tool_calls")
            if tool_calls:
                names = [tc.get("function", {}).get("name", "?") for tc in tool_calls]
                content_str = f"[called tools: {', '.join(names)}] {content_str}".strip()
            lines.append(f"{role}: {content_str}")

        summary_max_tokens = min(800, max(200, len(old_messages) * 60))
        summary_prompt = [
            {
                "role": "user",
                "content": (
                    "Summarize the following agent reasoning steps, tool calls, and results. "
                    "Preserve key facts, decisions, tool outcomes, errors encountered, and any "
                    "information needed to continue the task. Be concise but complete:\n\n"
                    + "\n".join(lines)
                ),
            }
        ]

        try:
            result = await self.llm.chat_completion(
                messages=summary_prompt,
                max_tokens=summary_max_tokens,
                temperature=0.3,
            )
            summary_text = result["choices"][0]["message"]["content"].strip()
        except Exception as e:
            logger.warning(f"[AgentLoop] Context compaction LLM call failed: {e}")
            return messages

        compacted = (
            system_prefix
            + [
                {
                    "role": "system",
                    "content": f"[Agent Reasoning Summary — earlier steps compacted]: {summary_text}",
                }
            ]
            + recent_messages
        )
        logger.info(
            f"[AgentLoop] Context compacted: {len(messages)} → {len(compacted)} messages "
            f"({len(old_messages)} old turns summarized, summary budget: {summary_max_tokens} tokens)"
        )
        return compacted

    # ------------------------------------------------------------------
    # Tool schema loading
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # System prompt assembly
    # ------------------------------------------------------------------

    def _build_system_prompt(self, input_data: dict, config: dict) -> str:
        """Build system prompt from upstream context fields."""
        parts = []
        if config.get("include_plan_in_context", True):
            v = input_data.get("plan_context")
            if v:
                parts.append(v)
        if config.get("include_memory_context", True):
            v = input_data.get("_memory_context")
            if v:
                parts.append(f"## User Memories\n{v}")
        if config.get("include_knowledge_context", True):
            v = input_data.get("knowledge_context")
            if v:
                parts.append(f"## Relevant Knowledge\n{v}")
        if config.get("include_reasoning_context", True):
            v = input_data.get("reasoning_context")
            if v:
                parts.append(f"## Previous Reasoning\n{v}")
        return "\n\n".join(parts) if parts else ""

    # ------------------------------------------------------------------
    # LLM retry wrapper
    # ------------------------------------------------------------------

    async def _llm_with_retry(
        self,
        messages: list,
        model: str,
        temperature: float,
        max_tokens: int,
        tools: list,
        max_retries: int,
        retry_delay: float,
    ) -> dict:
        """Call LLM with exponential backoff retry on failure.

        Retries when the response is None/empty, contains an 'error' key, or
        is missing 'choices'.  Returns None after all retries are exhausted.
        """
        last_error = "Unknown error"
        for attempt in range(max_retries + 1):
            if attempt > 0:
                await asyncio.sleep(retry_delay * (2 ** (attempt - 1)))
            try:
                response = await self.llm.chat_completion(
                    messages=messages,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    tools=tools if tools else None,
                )
                if response and "choices" in response and not response.get("error"):
                    return response
                last_error = (
                    response.get("error", "Response missing 'choices' key")
                    if response
                    else "LLM returned None"
                )
            except Exception as e:
                last_error = str(e)
        logger.warning(
            f"[{self.__class__.__name__}] LLM failed after {max_retries + 1} attempts: {last_error}"
        )
        return None

    # ------------------------------------------------------------------
    # Session-manager logging helpers
    # ------------------------------------------------------------------

    def _log_tool_call(self, func_name: str, args: dict) -> float:
        """Log a tool call to the session trace and return start_time."""
        start_time = time.time()
        if self._session_manager:
            try:
                self._session_manager.log_tool_call(func_name, args)
            except Exception:
                pass
        return start_time

    def _log_tool_result(
        self,
        func_name: str,
        output: object,
        success: bool,
        start_time: float,
        error: str = None,
    ) -> None:
        """Log a tool result to the session trace."""
        if self._session_manager:
            try:
                duration_ms = (time.time() - start_time) * 1000
                self._session_manager.log_tool_result(
                    func_name,
                    output,
                    success=success,
                    duration_ms=duration_ms,
                    error=error,
                )
            except Exception:
                pass

    def _log_llm_call(self, model: str) -> float:
        """Log an LLM call to the session trace and return start_time."""
        start_time = time.time()
        if self._session_manager:
            try:
                self._session_manager.log_llm_call(model, tokens=None)
            except Exception:
                pass
        return start_time

    def _log_llm_result(self, model: str, start_time: float, response: dict) -> None:
        """Log LLM latency + token usage to the session trace."""
        if self._session_manager:
            try:
                latency_ms = (time.time() - start_time) * 1000
                tokens = None
                if response and "usage" in response:
                    tokens = response["usage"].get("total_tokens")
                self._session_manager.log_llm_call(
                    model, tokens=tokens, latency_ms=latency_ms
                )
            except Exception:
                pass

    def _log_agent_event(self, event_type: str, data: dict = None) -> None:
        """Log a general agent lifecycle event to the session trace."""
        if self._session_manager:
            try:
                self._session_manager.log_agent_event(event_type, data)
            except Exception:
                pass

    def _log_rlm_event(self, event_type: str, data: dict = None) -> None:
        """Log an RLM-specific event to the session trace."""
        if self._session_manager:
            try:
                self._session_manager.log_rlm_event(event_type, data)
            except Exception:
                pass

    def _log_replan_event(self, result: dict) -> None:
        """Log a replan event when replan_needed is True, or agent_end on success."""
        if result.get("replan_needed"):
            self._log_agent_event("replan", {
                "reason": result.get("replan_reason"),
                "replan_count": result.get("replan_count"),
                "suggested_approach": result.get("suggested_approach"),
                "agent_loop_error": result.get("agent_loop_error"),
            })
        else:
            self._log_agent_event("agent_end", {
                "iterations": result.get("iterations", 0),
                "replan_count": result.get("replan_count", 0),
            })

    # ------------------------------------------------------------------
    # Shared REPL helpers — used by REPLEnvironmentExecutor and RLM/Hybrid
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_user_content(messages: list) -> str:
        """Extract first non-empty user message content from a message list."""
        for msg in messages:
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if content:
                    return content
        return " ".join(m.get("content", "") for m in messages if m.get("role") == "user")

    @staticmethod
    def _classify_content(content: str) -> str:
        """Classify content type for REPL environment metadata."""
        content_lower = content.lower()
        if "code" in content_lower or "def " in content_lower or "function" in content_lower:
            return "code"
        elif len(content) > 100_000:
            return "large_document"
        elif len(content) > 10_000:
            return "long_text"
        return "standard"

    # ------------------------------------------------------------------
    # Async SubCall — shared by RLM and Hybrid executors
    # ------------------------------------------------------------------

    async def _execute_sub_call(self, args: dict, repl_state: dict) -> dict:
        """Execute SubCall as an async built-in (not via sandbox exec).

        Cannot run inside the sandbox because it requires an awaited LLM call.
        Guardrails (max_sub_calls, max_recursion_depth, max_cost_usd) are read
        from *repl_state* so each executor can configure its own limits.
        """
        if repl_state.get("sub_call_count", 0) >= repl_state.get("max_sub_calls", 50):
            return {
                "tool_call_id": "",
                "role": "tool",
                "name": "sub_call",
                "content": "Error: max_sub_calls limit reached",
                "success": False,
            }
        if repl_state.get("recursion_depth", 0) >= repl_state.get("max_recursion_depth", 3):
            return {
                "tool_call_id": "",
                "role": "tool",
                "name": "sub_call",
                "content": "Error: max_recursion_depth limit reached",
                "success": False,
            }
        if repl_state.get("estimated_cost", 0.0) >= repl_state.get("max_cost_usd", 1.0):
            return {
                "tool_call_id": "",
                "role": "tool",
                "name": "sub_call",
                "content": "Error: max_cost_usd limit reached",
                "success": False,
            }

        prompt = args.get("prompt", "")
        if not prompt:
            return {
                "tool_call_id": "",
                "role": "tool",
                "name": "sub_call",
                "content": "Error: No prompt provided for SubCall",
                "success": False,
            }

        model = (
            args.get("model")
            or repl_state.get("sub_call_model")
            or settings.get("default_model")
        )
        max_tokens = int(args.get("max_tokens", 2000))

        repl_state["sub_call_count"] = repl_state.get("sub_call_count", 0) + 1
        old_depth = repl_state.get("recursion_depth", 0)
        repl_state["recursion_depth"] = old_depth + 1
        try:
            response = await self.llm.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                model=model,
                max_tokens=max_tokens,
            )
            repl_state["recursion_depth"] = old_depth
            if response and "choices" in response:
                content = response["choices"][0]["message"].get("content", "")
                estimated_tokens = len(prompt) // 4 + max_tokens
                repl_state["estimated_cost"] = (
                    repl_state.get("estimated_cost", 0.0)
                    + (estimated_tokens / 1000) * 0.001
                )
                return {
                    "tool_call_id": "",
                    "role": "tool",
                    "name": "sub_call",
                    "content": content,
                    "success": True,
                    "model_used": model,
                    "sub_call_count": repl_state["sub_call_count"],
                    "estimated_cost": repl_state["estimated_cost"],
                }
            return {
                "tool_call_id": "",
                "role": "tool",
                "name": "sub_call",
                "content": (
                    f"Error in SubCall: "
                    f"{response.get('error', 'Unknown error') if response else 'No response'}"
                ),
                "success": False,
            }
        except Exception as e:
            repl_state["recursion_depth"] = old_depth
            return {
                "tool_call_id": "",
                "role": "tool",
                "name": "sub_call",
                "content": f"Error in SubCall: {e}",
                "success": False,
            }


# ---------------------------------------------------------------------------
# Standard Agent Loop
# ---------------------------------------------------------------------------

class AgentLoopExecutor(AgentBaseExecutor):
    # Class-level library cache: per-subclass so each executor loads its own tools
    _library_cache: dict = {"mtime": 0.0, "data": {}, "file_mtimes": {}}

    def _load_tool_library(self):
        """Load tool implementations from library directory with file-level mtime caching."""
        try:
            if os.path.exists(LIBRARY_DIR):
                current_files = {}
                for filename in os.listdir(LIBRARY_DIR):
                    if filename.endswith(".py"):
                        tool_name = filename[:-3]
                        code_path = os.path.join(LIBRARY_DIR, filename)
                        try:
                            current_files[tool_name] = os.path.getmtime(code_path)
                        except OSError:
                            pass

                cache_data = self.__class__._library_cache.get("data", {})
                cache_mtimes = self.__class__._library_cache.get("file_mtimes", {})

                needs_reload = (
                    set(current_files) != set(cache_mtimes)
                    or any(current_files[n] != cache_mtimes.get(n) for n in current_files)
                )

                if needs_reload or not cache_data:
                    library = {}
                    for filename in os.listdir(LIBRARY_DIR):
                        if filename.endswith(".py"):
                            tool_name = filename[:-3]
                            code_path = os.path.join(LIBRARY_DIR, filename)
                            with open(code_path, "r", encoding="utf-8") as f:
                                library[tool_name] = f.read()
                    self.__class__._library_cache["data"] = library
                    self.__class__._library_cache["file_mtimes"] = current_files
                return self.__class__._library_cache.get("data", {})
        except OSError as e:
            logger.warning(f"Failed to load tool library: {e}")
        return {}

    async def _execute_tool(self, tool_call: dict, tool_library: dict) -> dict:
        """Execute a single tool call in the sandbox and return a tool result message."""
        func_name = tool_call["function"]["name"]
        try:
            args = json.loads(tool_call["function"]["arguments"])
        except (json.JSONDecodeError, KeyError):
            args = {}

        start_time = self._log_tool_call(func_name, args)
        success = False
        if func_name in tool_library:
            code = tool_library[func_name]
            try:
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(
                    None,
                    self._sandbox.execute,
                    code,
                    {"args": args},
                )
                output = result.get("result", "Success (no result returned)")
                success = True
            except Exception as e:
                output = f"Error executing tool {func_name}: {str(e)}"
        else:
            output = f"Error: Tool {func_name} not found in library."

        self._log_tool_result(func_name, output, success, start_time)

        return {
            "tool_call_id": tool_call.get("id", ""),
            "role": "tool",
            "name": func_name,
            "content": str(output),
            "success": success,
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
        trace: list,
        stream_queue: asyncio.Queue = None,
        compact_threshold: int = 0,
        compact_keep_last: int = 6,
        max_tool_output_chars: int = 0,
        thinking_steps: list = None,
    ) -> tuple:
        """Core agent loop: LLM ↔ Tool execution until no more tool calls or max_iterations.

        Returns (final_response, iterations, had_tool_error, accumulated_usage).
        accumulated_usage is a dict with prompt_tokens, completion_tokens, total_tokens
        summed across all LLM calls in this loop run.
        """
        # Fix 2: use local thinking_steps instead of instance-level state
        if thinking_steps is None:
            thinking_steps = []

        iterations = 0
        final_response = None
        had_tool_error = False
        # Accumulate token usage across all LLM iterations
        accumulated_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        # Only compact again once enough new messages have accumulated since
        # the last compaction, preventing a no-op re-compaction every iteration.
        _compacted_at_len = 0

        for iteration in range(max_iterations):
            # Reset per-iteration: dedup catches duplicate calls within a single
            # LLM response; max_iterations bounds cross-iteration repetition.
            seen_calls: set = set()
            iterations += 1
            iteration_trace = {"iteration": iterations, "tool_calls": [], "errors": []}

            if compact_threshold > 0:
                new_since_compact = len(llm_messages) - _compacted_at_len
                if new_since_compact >= compact_keep_last:
                    current_tokens = self._estimate_messages_tokens(llm_messages)
                    if current_tokens > compact_threshold:
                        llm_messages = await self._compact_messages(
                            llm_messages, compact_keep_last
                        )
                        _compacted_at_len = len(llm_messages)

            llm_start = self._log_llm_call(model)
            response = await self._llm_with_retry(
                messages=llm_messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=tools_list,
                max_retries=max_llm_retries,
                retry_delay=retry_delay,
            )
            self._log_llm_result(model, llm_start, response)

            if not response or "choices" not in response:
                error_msg = (
                    response.get("error", "LLM returned no choices")
                    if response
                    else "LLM returned None"
                )
                iteration_trace["errors"].append(error_msg)
                trace.append(iteration_trace)
                break

            final_response = response
            # Accumulate token usage from each LLM response
            iter_usage = response.get("usage", {})
            if isinstance(iter_usage, dict):
                accumulated_usage["prompt_tokens"] += iter_usage.get("prompt_tokens", 0)
                accumulated_usage["completion_tokens"] += iter_usage.get("completion_tokens", 0)
                accumulated_usage["total_tokens"] += iter_usage.get("total_tokens", 0)
            assistant_message = response["choices"][0]["message"]

            tool_calls = assistant_message.get("tool_calls", [])
            if tool_calls:
                for tc in tool_calls:
                    fn = tc.get("function", {})
                    name = fn.get("name", "?")
                    args = fn.get("arguments", "{}")
                    try:
                        args_dict = json.loads(args)
                        args_display = ", ".join(
                            f"{k}={repr(v)[:80]}" for k, v in args_dict.items()
                        )
                    except Exception:
                        args_display = args[:160]
                    thinking_steps.append(
                        {"type": "tool_call", "name": name, "content": f"{name}({args_display})"}
                    )
                if stream_queue:
                    await stream_queue.put(
                        {"type": "thinking", "content": list(thinking_steps)}
                    )

            llm_messages.append(assistant_message)

            if not tool_calls:
                trace.append(iteration_trace)
                break

            tool_error_occurred = False
            for tool_call in tool_calls:
                tool_name = tool_call.get("function", {}).get("name", "unknown")
                tool_args = tool_call.get("function", {}).get("arguments", "{}")

                try:
                    args_dict = json.loads(tool_args)
                    normalized_args = json.dumps(args_dict, sort_keys=True)
                except (json.JSONDecodeError, TypeError):
                    normalized_args = tool_args
                call_signature = f"{tool_name}:{normalized_args}"
                if call_signature in seen_calls:
                    iteration_trace["errors"].append(
                        f"Duplicate tool call skipped: '{tool_name}' with identical args in same response"
                    )
                    break
                seen_calls.add(call_signature)

                iteration_trace["tool_calls"].append(tool_name)
                tool_result = await self._execute_tool(tool_call, tool_library)

                # Fix 1: truncate large tool outputs at the source
                if max_tool_output_chars > 0:
                    raw = tool_result.get("content", "")
                    if len(raw) > max_tool_output_chars:
                        tool_result["content"] = (
                            f"[Output truncated: {len(raw):,} chars total, "
                            f"showing first {max_tool_output_chars:,}]\n"
                            + raw[:max_tool_output_chars]
                        )

                success = tool_result.get(
                    "success", False if not tool_result.get("content") else True
                )
                display = (tool_result.get("content") or "").strip()[:400]
                thinking_steps.append(
                    {"type": "tool_result", "name": tool_name, "content": display, "success": success}
                )
                if stream_queue:
                    await stream_queue.put(
                        {"type": "thinking", "content": list(thinking_steps)}
                    )

                if not tool_result.get("success", False):
                    had_tool_error = True
                    tool_error_occurred = True
                    iteration_trace["errors"].append(
                        f"Tool '{tool_name}': {tool_result['content']}"
                    )

                llm_messages.append(tool_result)

            trace.append(iteration_trace)

            if tool_error_occurred and tool_error_strategy == "stop":
                break

        return final_response, iterations, had_tool_error, accumulated_usage

    async def receive(self, input_data: dict, config: dict = None) -> dict:
        """
        Agent loop: repeatedly calls LLM with tools until no more tool calls.

        Input:
            - messages: conversation history (must include at least one user message)
            - Optional context keys: plan_context, _memory_context,
              knowledge_context, reasoning_context

        Config:
            - max_iterations (int, default 10)
            - max_tokens (int, default 2048)
            - temperature (float, default 0.7)
            - max_llm_retries (int, default 3)
            - retry_delay (float, default 1.0)
            - tool_error_strategy (str, default "continue"): "continue" or "stop"
            - timeout (float, default 120): 0 = disabled
            - max_replan_depth (int, default 3)
            - include_plan_in_context (bool, default True)
            - include_memory_context (bool, default True)
            - include_knowledge_context (bool, default True)
            - include_reasoning_context (bool, default True)

        Output:
            - messages, response, content, iterations, agent_loop_trace,
              agent_loop_error, replan_needed, replan_reason, suggested_approach,
              replan_count, episode_id (when episode persistence is enabled)
        """
        if input_data is None:
            input_data = {}
        config = config or {}

        # --- Session initialization ---
        try:
            sm = get_session_manager()
            sm.load_or_create_session()
            sm.log_agent_event("agent_start", {
                "input_keys": list(input_data.keys()),
                "config_keys": list(config.keys()),
            })
        except Exception as e:
            logger.warning(f"Failed to initialize session manager: {e}")
            sm = None

        # --- Read configuration ---
        max_iterations = int(config.get("max_iterations", 10))
        max_tokens = int(config.get("max_tokens", 2048))
        temperature = float(config.get("temperature", 0.7))
        max_llm_retries = int(config.get("max_llm_retries", 3))
        retry_delay = float(config.get("retry_delay", 1.0))
        tool_error_strategy = str(config.get("tool_error_strategy", "continue"))
        timeout = float(config.get("timeout", 120))
        max_replan_depth = int(config.get("max_replan_depth", 3))
        max_context_tokens = int(config.get("max_context_tokens", 6000))
        compact_threshold = int(config.get("compact_threshold", 0))
        compact_keep_last = int(config.get("compact_keep_last", 6))
        # Fix 1: read max_tool_output_chars from config
        max_tool_output_chars = int(config.get("max_tool_output_chars", 0))

        # --- Re-planning depth tracking ---
        replan_count = int(input_data.get("replan_count", 0))

        # --- Episode Persistence Support ---
        enable_episode = config.get("enable_episode_persistence", False)
        episode_id = config.get("episode_id")
        checkpoint_interval = config.get("checkpoint_interval", 1)
        if checkpoint_interval <= 0:
            logger.warning("checkpoint_interval must be >= 1, defaulting to 1")
            checkpoint_interval = 1

        episode = None
        iteration_count = 0

        if enable_episode and sm:
            try:
                existing_episode = sm.load_episode_state()
                should_resume = (
                    existing_episode
                    and existing_episode.phase not in [
                        EpisodeState.PHASE_COMPLETED,
                        EpisodeState.PHASE_FAILED,
                    ]
                    and (episode_id is None or existing_episode.episode_id == episode_id)
                )
                if should_resume:
                    episode = existing_episode
                    replan_count = episode.replan_count
                    input_data.setdefault("plan", episode.plan)
                    input_data.setdefault("completed_steps", episode.completed_steps)
                    input_data.setdefault("current_step", episode.current_step)
                    iteration_count = len(episode.checkpoints)
                    if episode.messages and config.get("resume_from_episode_messages", True):
                        input_data.setdefault("messages", episode.messages)
                    episode.update_phase(EpisodeState.PHASE_EXECUTING)
                    logger.info(f"Resuming episode {episode.episode_id}")
                elif episode_id or config.get("auto_create_episode", True):
                    budgets = {
                        "max_iterations": max_iterations,
                        "max_replan_depth": max_replan_depth,
                        "timeout": timeout,
                        "max_context_tokens": max_context_tokens,
                    }
                    episode = sm.create_episode(
                        input_data={"initial_input": list(input_data.keys())},
                        budgets=budgets,
                        metadata={"episode_id": episode_id} if episode_id else {},
                    )
                    logger.info(f"Created new episode {episode.episode_id}")
            except Exception as e:
                logger.warning(f"Failed to load episode state: {e}")

        # --- Re-planning depth guard ---
        if max_replan_depth > 0 and replan_count >= max_replan_depth:
            result = input_data.copy()
            result["replan_needed"] = False
            result["replan_depth_exceeded"] = True
            result["agent_loop_error"] = (
                f"Max re-planning depth ({max_replan_depth}) exceeded. Unable to complete task."
            )
            result["content"] = (
                f"[Error: Task failed after {replan_count} re-planning attempts. "
                "Unable to find viable execution path.]"
            )
            result["iterations"] = 0
            result["agent_loop_trace"] = []
            return result

        messages = input_data.get("messages", [])
        if not messages:
            return input_data

        system_prompt = self._build_system_prompt(input_data, config)
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
        agent_loop_trace: list = []

        # Fix 2: create thinking_steps locally before _execute()
        thinking_steps: list = []

        async def _execute():
            nonlocal iteration_count

            llm_messages = messages.copy()
            if max_context_tokens > 0:
                llm_messages = self._truncate_messages(llm_messages, max_context_tokens)

            # Inject context into system prompt without clobbering existing content
            if system_prompt:
                has_system = any(m.get("role") == "system" for m in llm_messages)
                if has_system:
                    context_markers = [
                        "## Execution Plan", "## User Memories",
                        "## Relevant Knowledge", "## Previous Reasoning",
                    ]
                    sys_content = next(
                        m.get("content", "") for m in llm_messages if m.get("role") == "system"
                    )
                    if not any(marker in sys_content for marker in context_markers):
                        for m in llm_messages:
                            if m.get("role") == "system":
                                m["content"] = m["content"] + "\n\n" + system_prompt
                                break
                else:
                    llm_messages.insert(0, {"role": "system", "content": system_prompt})

            final_response, iterations, had_tool_error, accumulated_usage = await self._run_agent_loop(
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
                trace=agent_loop_trace,
                stream_queue=config.get("_stream_queue"),
                compact_threshold=compact_threshold,
                compact_keep_last=compact_keep_last,
                max_tool_output_chars=max_tool_output_chars,
                thinking_steps=thinking_steps,
            )

            # Episode checkpoint

            if episode is not None and sm is not None:
                iteration_count += iterations
                if iteration_count % checkpoint_interval == 0 or iterations == 0:
                    try:
                        sm.save_episode_state(
                            phase=EpisodeState.PHASE_EXECUTING,
                            replan_count=replan_count,
                            completed_steps=input_data.get("completed_steps", []),
                            current_step=input_data.get("current_step", 0),
                            plan=input_data.get("plan", []),
                            messages=llm_messages if config.get("save_messages_in_episode", False) else None,
                            add_checkpoint=True,
                        )
                    except Exception as e:
                        logger.warning(f"Failed to save episode checkpoint: {e}")

            result = input_data.copy()
            result["messages"] = llm_messages
            result["response"] = final_response
            result["iterations"] = iterations
            # Expose accumulated token usage so the chat router can report it to the UI
            if accumulated_usage.get("total_tokens", 0) > 0:
                result["usage"] = accumulated_usage

            if final_response and "choices" in final_response:
                result["content"] = final_response["choices"][0]["message"].get("content", "")


            # Re-planning detection — use structured flags, not text matching
            content = result.get("content", "")
            has_error_flag = final_response.get("error") is not None if final_response else False
            actually_failed = had_tool_error or not content or has_error_flag

            plan = input_data.get("plan", [])
            current_step = input_data.get("current_step", 0)

            if actually_failed:
                result["replan_needed"] = True
                result["replan_count"] = replan_count + 1
                if had_tool_error:
                    result["replan_reason"] = "Tool execution errors occurred during plan execution"
                elif not content:
                    result["replan_reason"] = "Agent produced no content response"
                else:
                    result["replan_reason"] = "Agent response contained error indicators"
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
                    result["suggested_approach"] = (
                        "No plan exists. Consider creating a step-by-step plan."
                    )
            else:
                result["replan_needed"] = False
                result["replan_count"] = 0

            self._log_replan_event(result)

            # Episode finalization
            if episode is not None and sm is not None:
                try:
                    if result.get("agent_loop_error"):
                        final_phase = EpisodeState.PHASE_FAILED
                    elif result.get("replan_needed"):
                        final_phase = EpisodeState.PHASE_REPLANNING
                    else:
                        final_phase = EpisodeState.PHASE_COMPLETED
                    sm.save_episode_state(
                        phase=final_phase,
                        replan_count=result.get("replan_count", 0),
                        completed_steps=result.get("completed_steps", []),
                        current_step=result.get("current_step", 0),
                        plan=result.get("plan", []),
                        messages=llm_messages if config.get("save_messages_in_episode", False) else None,
                        add_checkpoint=True,
                    )
                    result["episode_id"] = episode.episode_id
                except Exception as e:
                    logger.warning(f"Failed to save final episode state: {e}")

            return result

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
            plan = input_data.get("plan", [])
            result["replan_needed"] = True
            result["replan_count"] = replan_count + 1
            result["replan_reason"] = f"Execution timed out after {timeout} seconds"
            result["suggested_approach"] = (
                f"Plan with {len(plan)} steps took too long. "
                "Consider breaking into smaller chunks with shorter timeout."
                if plan and len(plan) > 1
                else "Execution timed out. Consider simplifying the approach."
            )
            self._thinking_steps = thinking_steps
            self._log_replan_event(result)
            return result
        except Exception as e:
            result = input_data.copy()
            result["agent_loop_error"] = str(e)
            result["iterations"] = 0
            result["agent_loop_trace"] = agent_loop_trace
            result["replan_needed"] = True
            result["replan_count"] = replan_count + 1
            result["replan_reason"] = f"Unexpected error: {str(e)}"
            result["suggested_approach"] = "Review plan and try alternative approach."
            self._thinking_steps = thinking_steps
            self._log_replan_event(result)
            return result

        result["agent_loop_trace"] = agent_loop_trace
        # Fix 2: assign thinking_steps to instance attr at the end of all exit paths
        self._thinking_steps = thinking_steps
        return result

    async def send(self, processed_data: dict) -> dict:
        return processed_data


async def get_executor_class(node_type_id: str):
    if node_type_id == "agent_loop":
        return AgentLoopExecutor
    if node_type_id == "recursive_lm":
        return RLMAgentLoopExecutor
    if node_type_id == "repl_environment":
        return REPLEnvironmentExecutor
    if node_type_id == "hybrid_agent":
        return HybridAgentExecutor
    if node_type_id == "goal_pursuit":
        return GoalPursuitExecutor
    return None


# ---------------------------------------------------------------------------
# REPL Environment (setup node for RLM processing)
# ---------------------------------------------------------------------------

class REPLEnvironmentExecutor:
    """
    REPL Environment Node — initializes a persistent REPL state.

    1. Extracts user input from messages.
    2. Stores it as prompt_var in repl_state (NOT in context window).
    3. Builds a metadata-only system prompt.
    4. Initializes full repl_state including recursion tracking.

    Config:
        - max_recursion_depth (int, default 3)
        - max_cost_usd (float, default 1.0)
        - max_sub_calls (int, default 50)
        - sub_call_model (str): model for sub-calls (defaults to root model)
    """

    async def receive(self, input_data: dict, config: dict = None) -> dict:
        if input_data is None:
            return None

        config = config or {}
        messages = input_data.get("messages", [])

        user_content = AgentBaseExecutor._extract_user_content(messages)

        repl_state = {
            "prompt_var": user_content,
            "prompt_length": len(user_content),
            "variables": {},
            "stdout_history": [],
            "final": None,
            "iteration": 0,
            "recursion_depth": 0,
            "max_recursion_depth": config.get("max_recursion_depth", 3),
            "sub_call_count": 0,
            "max_sub_calls": config.get("max_sub_calls", 50),
            "estimated_cost": 0.0,
            "max_cost_usd": config.get("max_cost_usd", 1.0),
            "root_model": config.get("model") or config.get("root_model"),
            "sub_call_model": config.get("sub_call_model") or config.get("model"),
        }

        preview = user_content[:200] + "..." if len(user_content) > 200 else user_content
        system_content = self._build_repl_system_prompt({
            "prompt_length": len(user_content),
            "prompt_preview": preview,
            "prompt_type": AgentBaseExecutor._classify_content(user_content),
            "max_recursion_depth": repl_state["max_recursion_depth"],
            "max_sub_calls": repl_state["max_sub_calls"],
        })

        result = input_data.copy()
        result["repl_state"] = repl_state
        result["_repl_initialized"] = True
        result["messages"] = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": "Process the input using the REPL environment."},
        ]
        return result

    def _build_repl_system_prompt(self, metadata: dict) -> str:
        return f"""You are operating in a REPL environment.

The user's input has been loaded as a variable. DO NOT try to process it all at once.

Environment state:
- prompt_length: {metadata['prompt_length']} characters
- prompt_type: {metadata['prompt_type']}
- prompt_preview: "{metadata['prompt_preview']}"
- max_recursion_depth: {metadata['max_recursion_depth']}
- max_sub_calls: {metadata['max_sub_calls']}

Available functions:
- peek(start, end): View a slice of the prompt by character position
- search(pattern): Find regex matches in the prompt
- chunk(size, overlap): Split prompt into chunks of given size
- sub_call(prompt): Recursively call an LLM on any string
- set_variable(name, value): Store intermediate results
- get_variable(name): Retrieve stored results
- set_final(value): Set your final answer and terminate

Write Python code to examine, decompose, and process the input.
Store intermediate results in variables using set_variable().
Call set_final() when complete.

WARNING: Do not exceed max_recursion_depth or max_sub_calls or execution will be terminated."""

    async def send(self, processed_data: dict) -> dict:
        return processed_data


# ---------------------------------------------------------------------------
# RLM Agent Loop
# ---------------------------------------------------------------------------

class RLMAgentLoopExecutor(AgentBaseExecutor):
    """
    RLM (Recursive Language Model) variant of the agent loop.

    Key differences from standard AgentLoopExecutor:
    1. REPL state is maintained across iterations — input stored as variable, not in context.
    2. Tool outputs are stored as variables in repl_state, NOT injected into messages.
    3. Only metadata about stdout goes back into LLM history (constant size).
    4. Terminates when set_final() is called.

    This architecture enables processing arbitrarily long inputs by keeping the
    LLM context window at constant size regardless of input or output length.
    """

    _library_cache: dict = {"mtime": 0.0, "data": {}, "file_mtimes": {}}

    def _load_tool_library(self):
        """Load RLM tool implementations from both LIBRARY_DIR and RLM_LIBRARY_DIR.

        RLM tools take priority over standard tools on name collision.
        Uses per-file mtime fingerprints for accurate cache invalidation —
        catches content edits that don't update the directory mtime.
        """
        try:
            current_files: dict = {}
            for lib_dir in (LIBRARY_DIR, RLM_LIBRARY_DIR):
                if not os.path.exists(lib_dir):
                    continue
                for filename in os.listdir(lib_dir):
                    if filename.endswith(".py"):
                        tool_name = filename[:-3]
                        code_path = os.path.join(lib_dir, filename)
                        try:
                            current_files[tool_name] = os.path.getmtime(code_path)
                        except OSError:
                            pass

            cache_mtimes = self.__class__._library_cache.get("file_mtimes", {})
            needs_reload = (
                set(current_files) != set(cache_mtimes)
                or any(current_files[n] != cache_mtimes.get(n) for n in current_files)
            )

            if needs_reload or not self.__class__._library_cache.get("data"):
                library: dict = {}
                # Standard tools loaded first
                if os.path.exists(LIBRARY_DIR):
                    for filename in os.listdir(LIBRARY_DIR):
                        if filename.endswith(".py"):
                            tool_name = filename[:-3]
                            code_path = os.path.join(LIBRARY_DIR, filename)
                            with open(code_path, "r", encoding="utf-8") as f:
                                library[tool_name] = f.read()
                # RLM tools override standard tools on name collision
                if os.path.exists(RLM_LIBRARY_DIR):
                    for filename in os.listdir(RLM_LIBRARY_DIR):
                        if filename.endswith(".py"):
                            tool_name = filename[:-3]
                            code_path = os.path.join(RLM_LIBRARY_DIR, filename)
                            with open(code_path, "r", encoding="utf-8") as f:
                                library[tool_name] = f.read()
                self.__class__._library_cache["data"] = library
                self.__class__._library_cache["file_mtimes"] = current_files
            return self.__class__._library_cache.get("data", {})
        except OSError as e:
            logger.warning(f"Failed to load RLM tool library: {e}")
        return {}

    def _resolve_tool_name(self, func_name: str, tool_library: dict) -> str:
        """Resolve tool name with case-insensitive and camelCase↔snake_case matching."""
        if func_name in tool_library:
            return func_name

        func_lower = func_name.lower()
        for tool_name in tool_library:
            if tool_name.lower() == func_lower:
                return tool_name

        # snake_case → CamelCase
        camel = "".join(p.capitalize() for p in func_name.split("_"))
        if camel in tool_library:
            return camel

        # CamelCase → snake_case
        snake = ""
        for i, char in enumerate(func_name):
            if char.isupper() and i > 0:
                snake += "_"
            snake += char.lower()
        if snake in tool_library:
            return snake

        return func_name

    def _build_repl_system_prompt(self, metadata: dict) -> str:
        return f"""You are operating in a REPL environment.

The user's input has been loaded as a variable. DO NOT try to process it all at once.

Environment state:
- prompt_length: {metadata['prompt_length']} characters
- prompt_type: {metadata['prompt_type']}
- prompt_preview: "{metadata['prompt_preview']}"

Available functions:
- peek(start, end): View a slice of the prompt by character position
- search(pattern): Find regex matches in the prompt
- chunk(size, overlap): Split prompt into chunks of given size
- sub_call(prompt): Recursively call an LLM on any string
- set_variable(name, value): Store intermediate results
- get_variable(name): Retrieve stored results
- set_final(value): Set your final answer and terminate

Write Python code to examine, decompose, and process the input.
Store intermediate results in variables using set_variable().
Call set_final() when complete."""

    def _init_repl_state(self, messages: list) -> tuple:
        """Initialize REPL state from messages.

        Returns (repl_state, metadata, new_messages).
        """
        user_content = self._extract_user_content(messages)

        repl_state = {
            "prompt_var": user_content,
            "variables": {},
            "stdout_history": [],
            "final": None,
            "iteration": 0,
        }

        preview = user_content[:200] + "..." if len(user_content) > 200 else user_content
        metadata = {
            "prompt_length": len(user_content),
            "prompt_preview": preview,
            "prompt_type": self._classify_content(user_content),
        }

        system_content = self._build_repl_system_prompt(metadata)
        new_messages = [
            {"role": "system", "content": system_content},
            {
                "role": "user",
                "content": (
                    "Process the input using the REPL environment. "
                    "Use tools to examine the content incrementally and call set_final() when done."
                ),
            },
        ]
        return repl_state, metadata, new_messages

    async def _execute_tool(
        self, tool_call: dict, tool_library: dict, repl_state: dict
    ) -> dict:
        """Execute a tool call — SubCall is intercepted as an async built-in."""
        func_name = tool_call["function"]["name"]
        try:
            args = json.loads(tool_call["function"]["arguments"])
        except (json.JSONDecodeError, KeyError):
            args = {}

        # SubCall requires an async LLM call — cannot go through sandbox exec()
        if func_name.lower() in ("subcall", "sub_call"):
            self._log_rlm_event("sub_call", {"prompt_preview": str(args.get("prompt", ""))[:200]})
            result = await self._execute_sub_call(args, repl_state)
            self._log_rlm_event("sub_call_result", {
                "success": result.get("success", False),
                "model_used": result.get("model_used"),
                "sub_call_count": result.get("sub_call_count"),
            })
            return result

        actual_tool_name = self._resolve_tool_name(func_name, tool_library)
        # Strip _repl_state from log input (not useful in trace)
        log_args = {k: v for k, v in args.items() if k != "_repl_state"}
        start_time = self._log_tool_call(actual_tool_name, log_args)
        args["_repl_state"] = repl_state

        success = False
        if actual_tool_name in tool_library:
            code = tool_library[actual_tool_name]
            try:
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(
                    None,
                    self._sandbox.execute,
                    code,
                    {"args": args},
                )
                output = result.get("result", "Success (no result returned)")
                success = True
                if isinstance(result.get("_repl_state_update"), dict):
                    repl_state.update(result["_repl_state_update"])
                    # Log set_final when final answer is written via the REPL tool
                    if "final" in result["_repl_state_update"]:
                        self._log_rlm_event("set_final", {
                            "answer_length": len(str(result["_repl_state_update"]["final"])),
                        })
            except Exception as e:
                output = f"Error executing tool {func_name}: {str(e)}"
        else:
            output = f"Error: Tool {func_name} not found in library."

        self._log_tool_result(actual_tool_name, output, success, start_time)

        return {
            "tool_call_id": tool_call.get("id", ""),
            "role": "tool",
            "name": func_name,
            "content": str(output),
            "success": success,
        }

    async def _run_rlm_loop(
        self,
        initial_messages: list,
        repl_state: dict,
        tools_list: list,
        tool_library: dict,
        model: str,
        config: dict,
        trace: list,
        stream_queue: asyncio.Queue = None,
        thinking_steps: list = None,
    ) -> tuple:
        """Core RLM loop: LLM generates code → executes tools → only metadata goes back.

        Key difference from the standard loop: tool outputs are stored in
        repl_state["variables"], NOT injected into messages.  Terminates when
        repl_state["final"] is set via set_final().

        Returns (final_answer, final_repl_state, conversation_messages).
        """
        if thinking_steps is None:
            thinking_steps = []

        max_iterations = config.get("max_iterations", 20)
        stdout_preview_length = config.get("stdout_preview_length", 500)
        max_llm_retries = config.get("max_llm_retries", 3)
        retry_delay = config.get("retry_delay", 1.0)
        temperature = config.get("temperature", 0.1)
        max_tokens = config.get("max_tokens", 2000)

        llm_messages = initial_messages.copy()

        for iteration in range(max_iterations):
            repl_state["iteration"] = iteration
            iteration_trace = {"iteration": iteration, "tool_calls": [], "errors": []}

            state_metadata = {
                "iteration": iteration,
                "variables_set": list(repl_state["variables"].keys()),
                "stdout_entries": len(repl_state["stdout_history"]),
                "last_stdout_preview": (
                    repl_state["stdout_history"][-1][:stdout_preview_length]
                    if repl_state["stdout_history"]
                    else "none"
                ),
                "final_set": repl_state["final"] is not None,
            }

            messages_with_state = llm_messages + [
                {"role": "system", "content": f"REPL state: {json.dumps(state_metadata)}"}
            ]

            llm_start = self._log_llm_call(model)
            response = await self._llm_with_retry(
                messages=messages_with_state,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=tools_list,
                max_retries=max_llm_retries,
                retry_delay=retry_delay,
            )
            self._log_llm_result(model, llm_start, response)

            if not response or "choices" not in response:
                error_msg = (
                    response.get("error", "LLM returned no choices")
                    if response
                    else "LLM returned None"
                )
                iteration_trace["errors"].append(error_msg)
                trace.append(iteration_trace)
                break

            assistant_message = response["choices"][0]["message"]
            tool_calls = assistant_message.get("tool_calls", [])

            if tool_calls:
                for tc in tool_calls:
                    fn = tc.get("function", {})
                    name = fn.get("name", "?")
                    args_raw = fn.get("arguments", "{}")
                    try:
                        args_dict = json.loads(args_raw)
                        args_display = ", ".join(
                            f"{k}={repr(v)[:80]}" for k, v in args_dict.items()
                            if k != "_repl_state"
                        )
                    except Exception:
                        args_display = args_raw[:160]
                    thinking_steps.append(
                        {"type": "tool_call", "name": name, "content": f"{name}({args_display})"}
                    )
                if stream_queue:
                    await stream_queue.put(
                        {"type": "thinking", "content": list(thinking_steps)}
                    )

            if not tool_calls:
                content = assistant_message.get("content", "")
                if content:
                    repl_state["stdout_history"].append(content)
                    llm_messages.append({
                        "role": "assistant",
                        "content": f"[stdout: {len(content)} chars, preview: {content[:100]}...]",
                    })
                if repl_state.get("final") is not None:
                    break
                break

            for tool_call in tool_calls:
                tool_name = tool_call.get("function", {}).get("name", "unknown")
                iteration_trace["tool_calls"].append(tool_name)

                tool_result = await self._execute_tool(tool_call, tool_library, repl_state)

                output_content = tool_result["content"]
                repl_state["stdout_history"].append(str(output_content))

                # Only metadata enters LLM history — prevents context growth
                stdout_metadata = (
                    f"[stdout: {len(str(output_content))} chars, "
                    f"preview: {str(output_content)[:stdout_preview_length]}...]"
                )
                llm_messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.get("id", ""),
                    "content": stdout_metadata,
                })

                success = tool_result.get("success", False)
                display = str(output_content).strip()[:400]
                thinking_steps.append(
                    {"type": "tool_result", "name": tool_name, "content": display, "success": success}
                )
                if stream_queue:
                    await stream_queue.put(
                        {"type": "thinking", "content": list(thinking_steps)}
                    )

            if repl_state.get("final") is not None:
                trace.append(iteration_trace)
                break

            trace.append(iteration_trace)

        return repl_state.get("final"), repl_state, llm_messages

    async def receive(self, input_data: dict, config: dict = None) -> dict:
        """
        RLM Agent Loop: processes arbitrarily long inputs via REPL environment.

        Input:
            - messages: conversation history (must include at least one user message)

        Config:
            - max_iterations (int, default 20)
            - max_tokens (int, default 2000)
            - temperature (float, default 0.1): low temp for code generation
            - max_llm_retries (int, default 3)
            - retry_delay (float, default 1.0)
            - stdout_preview_length (int, default 500)
            - sub_call_model (str, optional)
            - model (str, optional)

        Output:
            - content: final answer from set_final()
            - repl_state: final REPL state summary (for debugging)
            - iterations: number of iterations executed
            - rlm_trace: per-iteration details
            - agent_loop_trace: same as rlm_trace (for API consistency)
        """
        if input_data is None:
            input_data = {}
        config = config or {}

        # --- Session initialization ---
        try:
            sm = get_session_manager()
            sm.load_or_create_session()
            sm.log_agent_event("agent_start", {
                "input_keys": list(input_data.keys()),
                "config_keys": list(config.keys()),
            })
        except Exception as e:
            logger.warning(f"[RLMAgent] Failed to initialize session manager: {e}")
            sm = None

        # --- Read configuration ---
        max_iterations = int(config.get("max_iterations", 20))
        max_tokens = int(config.get("max_tokens", 2000))
        temperature = float(config.get("temperature", 0.1))
        max_llm_retries = int(config.get("max_llm_retries", 3))
        retry_delay = float(config.get("retry_delay", 1.0))
        stdout_preview_length = int(config.get("stdout_preview_length", 500))
        model = config.get("model") or settings.get("default_model")
        timeout = float(config.get("timeout", 120))
        max_replan_depth = int(config.get("max_replan_depth", 3))
        enable_episode = config.get("enable_episode_persistence", False)
        episode_id = config.get("episode_id")
        checkpoint_interval = config.get("checkpoint_interval", 1)
        if checkpoint_interval <= 0:
            logger.warning("[RLMAgent] checkpoint_interval must be >= 1, defaulting to 1")
            checkpoint_interval = 1

        # --- Re-planning depth tracking ---
        replan_count = int(input_data.get("replan_count", 0))

        # --- Episode Persistence Support ---
        episode = None
        iteration_count = 0

        if enable_episode and sm:
            try:
                existing_episode = sm.load_episode_state()
                should_resume = (
                    existing_episode
                    and existing_episode.phase not in [
                        EpisodeState.PHASE_COMPLETED,
                        EpisodeState.PHASE_FAILED,
                    ]
                    and (episode_id is None or existing_episode.episode_id == episode_id)
                )
                if should_resume:
                    episode = existing_episode
                    replan_count = episode.replan_count
                    input_data.setdefault("plan", episode.plan)
                    input_data.setdefault("completed_steps", episode.completed_steps)
                    input_data.setdefault("current_step", episode.current_step)
                    iteration_count = len(episode.checkpoints)
                    if episode.messages and config.get("resume_from_episode_messages", True):
                        input_data.setdefault("messages", episode.messages)
                    episode.update_phase(EpisodeState.PHASE_EXECUTING)
                    logger.info(f"[RLMAgent] Resuming episode {episode.episode_id}")
                elif episode_id or config.get("auto_create_episode", True):
                    budgets = {
                        "max_iterations": max_iterations,
                        "max_replan_depth": max_replan_depth,
                        "timeout": timeout,
                    }
                    episode = sm.create_episode(
                        input_data={"initial_input": list(input_data.keys())},
                        budgets=budgets,
                        metadata={"episode_id": episode_id} if episode_id else {},
                    )
                    logger.info(f"[RLMAgent] Created new episode {episode.episode_id}")
            except Exception as e:
                logger.warning(f"[RLMAgent] Failed to load episode state: {e}")

        # --- Re-planning depth guard ---
        if max_replan_depth > 0 and replan_count >= max_replan_depth:
            result = input_data.copy()
            result["replan_needed"] = False
            result["replan_depth_exceeded"] = True
            result["agent_loop_error"] = (
                f"Max re-planning depth ({max_replan_depth}) exceeded. Unable to complete task."
            )
            result["content"] = (
                f"[Error: Task failed after {replan_count} re-planning attempts. "
                "Unable to find viable execution path.]"
            )
            result["iterations"] = 0
            result["rlm_trace"] = []
            result["agent_loop_trace"] = []
            return result

        messages = input_data.get("messages", [])
        if not messages:
            result = input_data.copy()
            result["content"] = ""
            result["rlm_error"] = "No messages provided"
            return result

        repl_state, metadata, new_messages = self._init_repl_state(messages)

        tools_def = self._load_tools()
        tools_list = []
        if tools_def:
            for tool_name, tool_data in tools_def.items():
                if isinstance(tool_data, dict) and tool_data.get("enabled", True):
                    definition = tool_data.get("definition")
                    if definition:
                        tools_list.append(definition)

        tool_library = self._load_tool_library()
        trace: list = []
        stream_queue = config.get("_stream_queue")
        # Local thinking_steps — assigned to self._thinking_steps at all exit paths
        thinking_steps: list = []

        async def _execute():
            nonlocal iteration_count

            final_result, final_state, conversation_messages = await self._run_rlm_loop(
                initial_messages=new_messages,
                repl_state=repl_state,
                tools_list=tools_list,
                tool_library=tool_library,
                model=model,
                config={
                    "max_iterations": max_iterations,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "max_llm_retries": max_llm_retries,
                    "retry_delay": retry_delay,
                    "stdout_preview_length": stdout_preview_length,
                },
                trace=trace,
                stream_queue=stream_queue,
                thinking_steps=thinking_steps,
            )

            # Episode checkpoint — use actual conversation messages, not stub
            if episode is not None and sm is not None:
                iterations = final_state.get("iteration", 0) + 1
                iteration_count += iterations
                if iteration_count % checkpoint_interval == 0 or iterations == 0:
                    try:
                        sm.save_episode_state(
                            phase=EpisodeState.PHASE_EXECUTING,
                            replan_count=replan_count,
                            completed_steps=input_data.get("completed_steps", []),
                            current_step=input_data.get("current_step", 0),
                            plan=input_data.get("plan", []),
                            messages=conversation_messages if config.get("save_messages_in_episode", False) else None,
                            add_checkpoint=True,
                        )
                    except Exception as e:
                        logger.warning(f"[RLMAgent] Failed to save episode checkpoint: {e}")

            result = input_data.copy()
            result["content"] = final_result or ""
            result["repl_state"] = {
                "variables": final_state.get("variables", {}),
                "iteration": final_state.get("iteration", 0),
                "stdout_count": len(final_state.get("stdout_history", [])),
            }
            result["iterations"] = final_state.get("iteration", 0) + 1
            result["rlm_trace"] = trace
            result["agent_loop_trace"] = trace
            result["messages"] = conversation_messages

            if not final_result:
                result["rlm_error"] = "No final answer set — max iterations reached or error"

            # Re-planning detection
            actually_failed = not final_result

            plan = input_data.get("plan", [])
            current_step = input_data.get("current_step", 0)

            if actually_failed:
                result["replan_needed"] = True
                result["replan_count"] = replan_count + 1
                result["replan_reason"] = (
                    "RLM agent produced no final answer (max iterations reached or set_final not called)"
                )
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
                    result["suggested_approach"] = (
                        "No plan exists. Consider creating a step-by-step plan."
                    )
            else:
                result["replan_needed"] = False
                result["replan_count"] = 0

            self._log_replan_event(result)

            # Episode finalization — use actual conversation messages
            if episode is not None and sm is not None:
                try:
                    if result.get("agent_loop_error"):
                        final_phase = EpisodeState.PHASE_FAILED
                    elif result.get("replan_needed"):
                        final_phase = EpisodeState.PHASE_REPLANNING
                    else:
                        final_phase = EpisodeState.PHASE_COMPLETED
                    sm.save_episode_state(
                        phase=final_phase,
                        replan_count=result.get("replan_count", 0),
                        completed_steps=result.get("completed_steps", []),
                        current_step=result.get("current_step", 0),
                        plan=result.get("plan", []),
                        messages=conversation_messages if config.get("save_messages_in_episode", False) else None,
                        add_checkpoint=True,
                    )
                    result["episode_id"] = episode.episode_id
                except Exception as e:
                    logger.warning(f"[RLMAgent] Failed to save final episode state: {e}")

            return result

        try:
            if timeout > 0:
                result = await asyncio.wait_for(_execute(), timeout=timeout)
            else:
                result = await _execute()
        except asyncio.TimeoutError:
            result = input_data.copy()
            result["agent_loop_error"] = f"RLM agent timed out after {timeout}s"
            result["iterations"] = 0
            result["rlm_trace"] = trace
            result["agent_loop_trace"] = trace
            plan = input_data.get("plan", [])
            result["replan_needed"] = True
            result["replan_count"] = replan_count + 1
            result["replan_reason"] = f"Execution timed out after {timeout} seconds"
            result["suggested_approach"] = (
                f"Plan with {len(plan)} steps took too long. "
                "Consider breaking into smaller chunks with shorter timeout."
                if plan and len(plan) > 1
                else "Execution timed out. Consider simplifying the approach."
            )
            self._thinking_steps = thinking_steps
            self._log_replan_event(result)
            return result
        except Exception as e:
            result = input_data.copy()
            result["agent_loop_error"] = str(e)
            result["iterations"] = 0
            result["rlm_trace"] = trace
            result["agent_loop_trace"] = trace
            result["replan_needed"] = True
            result["replan_count"] = replan_count + 1
            result["replan_reason"] = f"Unexpected error: {str(e)}"
            result["suggested_approach"] = "Review plan and try alternative approach."
            self._thinking_steps = thinking_steps
            self._log_replan_event(result)
            return result

        result["rlm_trace"] = trace
        result["agent_loop_trace"] = trace
        self._thinking_steps = thinking_steps
        return result

    async def send(self, processed_data: dict) -> dict:
        return processed_data


# ---------------------------------------------------------------------------
# Hybrid Agent Loop
# ---------------------------------------------------------------------------

class HybridAgentExecutor(AgentBaseExecutor):
    """
    Hybrid Agent Loop — standard tool calling with RLM-style adaptive output routing.

    Small tool outputs (≤ large_output_threshold chars) are placed inline in messages
    so the LLM can read them directly.  Large tool outputs are stored in
    repl_state["variables"] and only a compact metadata stub enters conversation
    history.  The LLM retrieves stored data on demand via GetVariable, Peek, Search,
    Chunk, or SetVariable.

    SetFinal is excluded because hybrid uses the no-tool-calls termination convention.
    All other RLM tools (Peek, Search, Chunk, GetVariable, SetVariable, SubCall) work
    alongside the full standard tool library.

    Advantages over the standard agent loop:
    - Context never grows unboundedly from tool outputs — no compaction needed.
    - Standard and RLM variable-access tools coexist in one tool set.
    - Handles tool outputs of arbitrary size without truncation.
    - Full re-planning and episode persistence support (mirrors AgentLoopExecutor).
    """

    _EXCLUDED_TOOLS: frozenset = frozenset({"SetFinal"})

    # Fix 3: retrieval tools that should never trigger adaptive routing
    _RETRIEVAL_TOOLS: frozenset = frozenset({"GetVariable", "Peek", "Search", "Chunk", "YouTubeTranscript"})

    _library_cache: dict = {"mtime": 0.0, "data": {}, "file_mtimes": {}}

    @staticmethod
    def _build_variable_system_note(threshold: int) -> str:
        """Return a brief system-level instruction explaining the variable storage system."""
        return (
            "## Variable Storage\n"
            f"Tool outputs longer than {threshold:,} characters are stored as named variables "
            "instead of being placed inline. When a tool result shows "
            "\"[Output stored as 'var_name': N chars | preview: ...]\", use these tools:\n"
            "- GetVariable(name) — read the full content of a variable\n"
            "- Peek(var_name=name, start=0, end=1000) — read a character-range slice\n"
            "- Search(var_name=name, pattern=\"regex\") — find regex matches inside a variable\n"
            "- Chunk(var_name=name, size=2000, overlap=200) — split into overlapping chunks\n"
            "- SetVariable(name, value) — store your own intermediate result\n"
            "- SubCall(prompt=\"...\") — run a focused sub-prompt through the LLM"
        )

    def _load_tool_library(self):
        """Load implementations from both LIBRARY_DIR and RLM_LIBRARY_DIR.

        Standard tools are loaded first; RLM tools override any name conflict so
        variable-access tools are always the RLM versions.  Uses file-level mtime
        fingerprints for accurate cache invalidation.
        """
        try:
            current_files: dict = {}
            for lib_dir in (LIBRARY_DIR, RLM_LIBRARY_DIR):
                if not os.path.exists(lib_dir):
                    continue
                for filename in os.listdir(lib_dir):
                    if filename.endswith(".py"):
                        tool_name = filename[:-3]
                        code_path = os.path.join(lib_dir, filename)
                        try:
                            current_files[tool_name] = os.path.getmtime(code_path)
                        except OSError:
                            pass

            cache_mtimes = self.__class__._library_cache.get("file_mtimes", {})
            needs_reload = (
                set(current_files) != set(cache_mtimes)
                or any(current_files[n] != cache_mtimes.get(n) for n in current_files)
            )

            if needs_reload or not self.__class__._library_cache.get("data"):
                library: dict = {}
                if os.path.exists(LIBRARY_DIR):
                    for filename in os.listdir(LIBRARY_DIR):
                        if filename.endswith(".py"):
                            tool_name = filename[:-3]
                            with open(os.path.join(LIBRARY_DIR, filename), "r", encoding="utf-8") as f:
                                library[tool_name] = f.read()
                # RLM tools override standard tools on name collision
                if os.path.exists(RLM_LIBRARY_DIR):
                    for filename in os.listdir(RLM_LIBRARY_DIR):
                        if filename.endswith(".py"):
                            tool_name = filename[:-3]
                            with open(os.path.join(RLM_LIBRARY_DIR, filename), "r", encoding="utf-8") as f:
                                library[tool_name] = f.read()
                self.__class__._library_cache["data"] = library
                self.__class__._library_cache["file_mtimes"] = current_files

            return self.__class__._library_cache.get("data", {})
        except OSError as e:
            logger.warning(f"[HybridAgent] Failed to load tool library: {e}")
        return {}

    def _variable_inventory_msg(self, repl_state: dict) -> dict | None:
        """Return an ephemeral system message listing currently stored variables.

        Injected before each LLM call but never persisted in llm_messages, so
        it adds zero tokens to the running context history.
        """
        variables = repl_state.get("variables", {})
        if not variables:
            return None
        lines = [f"  • {name}: {len(str(v)):,} chars" for name, v in variables.items()]
        return {
            "role": "system",
            "content": (
                "Stored variables (use GetVariable(name) to retrieve full content):\n"
                + "\n".join(lines)
            ),
        }

    async def _execute_tool(
        self,
        tool_call: dict,
        tool_library: dict,
        repl_state: dict,
        large_output_threshold: int,
    ) -> dict:
        """Execute one tool call with adaptive output routing.

        If the output exceeds *large_output_threshold* characters it is stored in
        repl_state["variables"] and a compact stub is returned to the LLM instead.
        The LLM can retrieve the full content via GetVariable("<var_name>").
        """
        func_name = tool_call["function"]["name"]
        try:
            args = json.loads(tool_call["function"]["arguments"])
        except (json.JSONDecodeError, KeyError):
            args = {}

        # SubCall requires an async LLM call — intercept before sandbox execution
        if func_name.lower() in ("subcall", "sub_call"):
            self._log_agent_event("hybrid_sub_call", {"prompt_preview": str(args.get("prompt", ""))[:200]})
            result = await self._execute_sub_call(args, repl_state)
            result["tool_call_id"] = tool_call.get("id", "")
            self._log_agent_event("hybrid_sub_call_result", {
                "success": result.get("success", False),
                "model_used": result.get("model_used"),
            })
            return result

        # Strip _repl_state from log args (internal, not useful in trace)
        log_args = {k: v for k, v in args.items() if k != "_repl_state"}
        start_time = self._log_tool_call(func_name, log_args)

        # Pass repl_state so RLM-library tools (Peek, GetVariable, …) work correctly
        args["_repl_state"] = repl_state

        success = False
        output = f"Error: Tool '{func_name}' not found in library."

        if func_name in tool_library:
            code = tool_library[func_name]
            try:
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(
                    None,
                    self._sandbox.execute,
                    code,
                    {"args": args},
                )
                output = result.get("result", "Success (no result returned)")
                success = True
                if isinstance(result.get("_repl_state_update"), dict):
                    repl_state.update(result["_repl_state_update"])
            except Exception as e:
                output = f"Error executing '{func_name}': {e}"

        output_str = str(output)
        self._log_tool_result(func_name, output_str, success, start_time)

        # Fix 3: adaptive routing only for non-retrieval tools
        # Retrieval tools (GetVariable, Peek, Search, Chunk) must always return inline
        # to avoid infinite indirection chains.
        if success and len(output_str) > large_output_threshold and func_name not in self._RETRIEVAL_TOOLS:
            count = repl_state["_var_counts"].get(func_name, 0) + 1
            repl_state["_var_counts"][func_name] = count
            var_name = f"var_{func_name.lower()}_{count}"
            repl_state["variables"][var_name] = output_str
            preview = output_str[:200].replace("\n", " ")
            content = (
                f"[Output stored as '{var_name}': {len(output_str):,} chars | "
                f"preview: {preview!r}]\n"
                f"Call GetVariable(\"{var_name}\") to read the full content."
            )
            self._log_agent_event("hybrid_variable_stored", {
                "tool": func_name,
                "var_name": var_name,
                "output_chars": len(output_str),
                "threshold": large_output_threshold,
            })
        else:
            content = output_str

        return {
            "tool_call_id": tool_call.get("id", ""),
            "role": "tool",
            "name": func_name,
            "content": content,
            "success": success,
        }

    async def _run_hybrid_loop(
        self,
        llm_messages: list,
        tools_list: list,
        tool_library: dict,
        repl_state: dict,
        model: str,
        temperature: float,
        max_tokens: int,
        max_iterations: int,
        max_llm_retries: int,
        retry_delay: float,
        tool_error_strategy: str,
        large_output_threshold: int,
        trace: list,
        stream_queue: asyncio.Queue = None,
        thinking_steps: list = None,
    ) -> tuple:
        """Core hybrid loop.  Returns (final_response, iterations, had_tool_error)."""
        # Fix 2: use local thinking_steps instead of instance-level state
        if thinking_steps is None:
            thinking_steps = []

        iterations = 0
        final_response = None
        had_tool_error = False

        for iteration in range(max_iterations):
            # Reset per-iteration: dedup catches duplicate calls within a single
            # LLM response; max_iterations bounds cross-iteration repetition.
            seen_calls: set = set()
            iterations += 1
            iteration_trace = {"iteration": iterations, "tool_calls": [], "errors": []}

            # Ephemeral variable inventory — injected per call, never stored in history
            var_msg = self._variable_inventory_msg(repl_state)
            messages_for_llm = llm_messages + ([var_msg] if var_msg else [])

            llm_start = self._log_llm_call(model)
            response = await self._llm_with_retry(
                messages=messages_for_llm,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=tools_list,
                max_retries=max_llm_retries,
                retry_delay=retry_delay,
            )
            self._log_llm_result(model, llm_start, response)

            if not response or "choices" not in response:
                error_msg = (
                    response.get("error", "LLM returned no choices")
                    if response
                    else "LLM returned None"
                )
                iteration_trace["errors"].append(error_msg)
                trace.append(iteration_trace)
                break

            final_response = response
            assistant_message = response["choices"][0]["message"]
            tool_calls = assistant_message.get("tool_calls", [])
            llm_messages.append(assistant_message)

            if tool_calls:
                for tc in tool_calls:
                    fn = tc.get("function", {})
                    name = fn.get("name", "?")
                    args_raw = fn.get("arguments", "{}")
                    try:
                        args_dict = json.loads(args_raw)
                        args_display = ", ".join(
                            f"{k}={repr(v)[:80]}"
                            for k, v in args_dict.items()
                            if k != "_repl_state"
                        )
                    except Exception:
                        args_display = args_raw[:160]
                    thinking_steps.append(
                        {"type": "tool_call", "name": name, "content": f"{name}({args_display})"}
                    )
                if stream_queue:
                    await stream_queue.put(
                        {"type": "thinking", "content": list(thinking_steps)}
                    )

            if not tool_calls:
                trace.append(iteration_trace)
                break

            tool_error_occurred = False
            for tool_call in tool_calls:
                tool_name = tool_call.get("function", {}).get("name", "unknown")
                tool_args = tool_call.get("function", {}).get("arguments", "{}")

                # Deduplication guard (per-iteration scope)
                try:
                    normalized_args = json.dumps(json.loads(tool_args), sort_keys=True)
                except (json.JSONDecodeError, TypeError):
                    normalized_args = tool_args
                call_sig = f"{tool_name}:{normalized_args}"
                if call_sig in seen_calls:
                    iteration_trace["errors"].append(
                        f"Duplicate tool call skipped: '{tool_name}' with identical args in same response"
                    )
                    break
                seen_calls.add(call_sig)

                iteration_trace["tool_calls"].append(tool_name)
                tool_result = await self._execute_tool(
                    tool_call, tool_library, repl_state, large_output_threshold
                )

                success = tool_result.get("success", False)
                display = (tool_result.get("content") or "").strip()[:400]
                thinking_steps.append(
                    {"type": "tool_result", "name": tool_name, "content": display, "success": success}
                )
                if stream_queue:
                    await stream_queue.put(
                        {"type": "thinking", "content": list(thinking_steps)}
                    )

                if not success:
                    had_tool_error = True
                    tool_error_occurred = True
                    iteration_trace["errors"].append(
                        f"Tool '{tool_name}': {tool_result['content']}"
                    )

                llm_messages.append(tool_result)

            trace.append(iteration_trace)

            if tool_error_occurred and tool_error_strategy == "stop":
                break

        return final_response, iterations, had_tool_error

    async def receive(self, input_data: dict, config: dict = None) -> dict:
        """Hybrid Agent Loop node executor.

        Combines standard tool calling with RLM-style adaptive output routing.
        Supports the same re-planning and episode persistence API as AgentLoopExecutor.

        Config:
            large_output_threshold (int, default 3000):
                Outputs longer than this many characters are stored as variables.
            max_iterations (int, default 15), max_tokens (int, default 4096),
            temperature (float, default 0.7), max_llm_retries (int, default 3),
            retry_delay (float, default 1.0), tool_error_strategy (str, default "continue"),
            timeout (float, default 120), max_replan_depth (int, default 3),
            enable_episode_persistence (bool, default False),
            include_plan_in_context / include_memory_context /
            include_knowledge_context / include_reasoning_context (bool, default True).

        Output keys include both hybrid_trace and agent_loop_trace (identical content)
        for compatibility with the standard agent loop API.
        """
        if input_data is None:
            input_data = {}
        config = config or {}

        # --- Session initialization ---
        try:
            sm = get_session_manager()
            sm.load_or_create_session()
            sm.log_agent_event("agent_start", {
                "input_keys": list(input_data.keys()),
                "config_keys": list(config.keys()),
            })
        except Exception as e:
            logger.warning(f"[HybridAgent] Failed to initialize session manager: {e}")
            sm = None

        # --- Read configuration ---
        max_iterations = int(config.get("max_iterations", 15))
        max_tokens = int(config.get("max_tokens", 4096))
        temperature = float(config.get("temperature", 0.7))
        max_llm_retries = int(config.get("max_llm_retries", 3))
        retry_delay = float(config.get("retry_delay", 1.0))
        tool_error_strategy = str(config.get("tool_error_strategy", "continue"))
        timeout = float(config.get("timeout", 120))
        max_replan_depth = int(config.get("max_replan_depth", 3))
        large_output_threshold = int(config.get("large_output_threshold", 3000))
        stream_queue = config.get("_stream_queue")
        model = config.get("model") or settings.get("default_model")

        # --- Re-planning depth tracking ---
        replan_count = int(input_data.get("replan_count", 0))

        # --- Episode Persistence Support ---
        enable_episode = config.get("enable_episode_persistence", False)
        episode_id = config.get("episode_id")
        checkpoint_interval = config.get("checkpoint_interval", 1)
        if checkpoint_interval <= 0:
            logger.warning("[HybridAgent] checkpoint_interval must be >= 1, defaulting to 1")
            checkpoint_interval = 1

        episode = None
        iteration_count = 0

        if enable_episode and sm:
            try:
                existing_episode = sm.load_episode_state()
                should_resume = (
                    existing_episode
                    and existing_episode.phase not in [
                        EpisodeState.PHASE_COMPLETED,
                        EpisodeState.PHASE_FAILED,
                    ]
                    and (episode_id is None or existing_episode.episode_id == episode_id)
                )
                if should_resume:
                    episode = existing_episode
                    replan_count = episode.replan_count
                    input_data.setdefault("plan", episode.plan)
                    input_data.setdefault("completed_steps", episode.completed_steps)
                    input_data.setdefault("current_step", episode.current_step)
                    iteration_count = len(episode.checkpoints)
                    if episode.messages and config.get("resume_from_episode_messages", True):
                        input_data.setdefault("messages", episode.messages)
                    episode.update_phase(EpisodeState.PHASE_EXECUTING)
                    logger.info(f"[HybridAgent] Resuming episode {episode.episode_id}")
                elif episode_id or config.get("auto_create_episode", True):
                    budgets = {
                        "max_iterations": max_iterations,
                        "max_replan_depth": max_replan_depth,
                        "timeout": timeout,
                    }
                    episode = sm.create_episode(
                        input_data={"initial_input": list(input_data.keys())},
                        budgets=budgets,
                        metadata={"episode_id": episode_id} if episode_id else {},
                    )
                    logger.info(f"[HybridAgent] Created new episode {episode.episode_id}")
            except Exception as e:
                logger.warning(f"[HybridAgent] Failed to load episode state: {e}")

        # --- Re-planning depth guard ---
        if max_replan_depth > 0 and replan_count >= max_replan_depth:
            result = input_data.copy()
            result["replan_needed"] = False
            result["replan_depth_exceeded"] = True
            result["agent_loop_error"] = (
                f"Max re-planning depth ({max_replan_depth}) exceeded. Unable to complete task."
            )
            result["content"] = (
                f"[Error: Task failed after {replan_count} re-planning attempts. "
                "Unable to find viable execution path.]"
            )
            result["iterations"] = 0
            result["agent_loop_trace"] = []
            result["hybrid_trace"] = []
            return result

        messages = input_data.get("messages", [])
        if not messages:
            result = input_data.copy()
            result["content"] = ""
            result["agent_loop_error"] = "No messages provided"
            return result

        # --- Build system prompt (context fields + variable storage note) ---
        system_prompt_parts = []
        context_prompt = self._build_system_prompt(input_data, config)
        if context_prompt:
            system_prompt_parts.append(context_prompt)
        system_prompt_parts.append(self._build_variable_system_note(large_output_threshold))
        system_prompt = "\n\n".join(system_prompt_parts)

        # --- Inject system prompt without clobbering existing system messages ---
        llm_messages = list(messages)
        has_system = any(m.get("role") == "system" for m in llm_messages)
        if has_system:
            context_markers = [
                "## Execution Plan", "## User Memories", "## Relevant Knowledge",
                "## Previous Reasoning", "## Variable Storage",
            ]
            sys_content = next(
                m.get("content", "") for m in llm_messages if m.get("role") == "system"
            )
            if not any(marker in sys_content for marker in context_markers):
                for m in llm_messages:
                    if m.get("role") == "system":
                        m["content"] = m["content"] + "\n\n" + system_prompt
                        break
        else:
            llm_messages.insert(0, {"role": "system", "content": system_prompt})

        # --- Load tools (exclude RLM-only tools that require prompt_var) ---
        tools_def = self._load_tools()
        tools_list = []
        for tool_name, tool_data in (tools_def or {}).items():
            if not isinstance(tool_data, dict) or not tool_data.get("enabled", True):
                continue
            if tool_name in self._EXCLUDED_TOOLS:
                continue
            definition = tool_data.get("definition")
            if definition:
                tools_list.append(definition)

        tool_library = self._load_tool_library()

        repl_state: dict = {
            "variables": {},
            "_var_counts": {},
            "sub_call_count": 0,
            "max_sub_calls": int(config.get("max_sub_calls", 20)),
            "recursion_depth": 0,
            "max_recursion_depth": int(config.get("max_recursion_depth", 3)),
            "estimated_cost": 0.0,
            "max_cost_usd": float(config.get("max_cost_usd", 1.0)),
            "sub_call_model": config.get("sub_call_model") or model,
        }
        trace: list = []
        # Fix 2: create thinking_steps locally before _execute()
        thinking_steps: list = []

        async def _execute():
            nonlocal iteration_count

            final_response, iterations, had_tool_error = await self._run_hybrid_loop(
                llm_messages=llm_messages,
                tools_list=tools_list,
                tool_library=tool_library,
                repl_state=repl_state,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                max_iterations=max_iterations,
                max_llm_retries=max_llm_retries,
                retry_delay=retry_delay,
                tool_error_strategy=tool_error_strategy,
                large_output_threshold=large_output_threshold,
                trace=trace,
                stream_queue=stream_queue,
                thinking_steps=thinking_steps,
            )

            # Episode checkpoint
            if episode is not None and sm is not None:
                iteration_count += iterations
                if iteration_count % checkpoint_interval == 0 or iterations == 0:
                    try:
                        sm.save_episode_state(
                            phase=EpisodeState.PHASE_EXECUTING,
                            replan_count=replan_count,
                            completed_steps=input_data.get("completed_steps", []),
                            current_step=input_data.get("current_step", 0),
                            plan=input_data.get("plan", []),
                            messages=llm_messages if config.get("save_messages_in_episode", False) else None,
                            add_checkpoint=True,
                        )
                    except Exception as e:
                        logger.warning(f"[HybridAgent] Failed to save episode checkpoint: {e}")

            content = ""
            if final_response and "choices" in final_response:
                content = final_response["choices"][0]["message"].get("content") or ""

            result = input_data.copy()
            result["messages"] = llm_messages
            result["content"] = content
            result["iterations"] = iterations
            result["hybrid_trace"] = trace
            result["agent_loop_trace"] = trace  # consistent with AgentLoopExecutor API
            result["repl_state"] = {
                "variables": list(repl_state["variables"].keys()),
                "variable_count": len(repl_state["variables"]),
            }

            # Re-planning detection (mirrors AgentLoopExecutor logic)
            has_error_flag = final_response.get("error") is not None if final_response else False
            actually_failed = had_tool_error or not content or has_error_flag

            plan = input_data.get("plan", [])
            current_step = input_data.get("current_step", 0)

            if actually_failed:
                result["replan_needed"] = True
                result["replan_count"] = replan_count + 1
                if had_tool_error:
                    result["replan_reason"] = "Tool execution errors occurred during plan execution"
                elif not content:
                    result["replan_reason"] = "Agent produced no content response"
                else:
                    result["replan_reason"] = "Agent response contained error indicators"
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
                    result["suggested_approach"] = (
                        "No plan exists. Consider creating a step-by-step plan."
                    )
            else:
                result["replan_needed"] = False
                result["replan_count"] = 0

            if had_tool_error:
                result["agent_loop_error"] = "One or more tools encountered errors"

            self._log_replan_event(result)

            # Episode finalization
            if episode is not None and sm is not None:
                try:
                    if result.get("agent_loop_error") and not had_tool_error:
                        final_phase = EpisodeState.PHASE_FAILED
                    elif result.get("replan_needed"):
                        final_phase = EpisodeState.PHASE_REPLANNING
                    else:
                        final_phase = EpisodeState.PHASE_COMPLETED
                    sm.save_episode_state(
                        phase=final_phase,
                        replan_count=result.get("replan_count", 0),
                        completed_steps=result.get("completed_steps", []),
                        current_step=result.get("current_step", 0),
                        plan=result.get("plan", []),
                        messages=llm_messages if config.get("save_messages_in_episode", False) else None,
                        add_checkpoint=True,
                    )
                    result["episode_id"] = episode.episode_id
                except Exception as e:
                    logger.warning(f"[HybridAgent] Failed to save final episode state: {e}")

            return result

        try:
            if timeout > 0:
                result = await asyncio.wait_for(_execute(), timeout=timeout)
            else:
                result = await _execute()
        except asyncio.TimeoutError:
            result = input_data.copy()
            result["messages"] = llm_messages
            result["content"] = ""
            result["agent_loop_error"] = f"Hybrid agent timed out after {timeout}s"
            result["iterations"] = 0
            result["hybrid_trace"] = trace
            result["agent_loop_trace"] = trace
            plan = input_data.get("plan", [])
            result["replan_needed"] = True
            result["replan_count"] = replan_count + 1
            result["replan_reason"] = f"Execution timed out after {timeout} seconds"
            result["suggested_approach"] = (
                f"Plan with {len(plan)} steps took too long. "
                "Consider breaking into smaller chunks with shorter timeout."
                if plan and len(plan) > 1
                else "Execution timed out. Consider simplifying the approach."
            )
            self._thinking_steps = thinking_steps
            self._log_replan_event(result)
            return result
        except Exception as e:
            result = input_data.copy()
            result["messages"] = llm_messages
            result["content"] = ""
            result["agent_loop_error"] = str(e)
            result["iterations"] = 0
            result["hybrid_trace"] = trace
            result["agent_loop_trace"] = trace
            result["replan_needed"] = True
            result["replan_count"] = replan_count + 1
            result["replan_reason"] = f"Unexpected error: {str(e)}"
            result["suggested_approach"] = "Review plan and try alternative approach."
            self._thinking_steps = thinking_steps
            self._log_replan_event(result)
            return result

        result["hybrid_trace"] = trace
        result["agent_loop_trace"] = trace
        # Fix 2: assign thinking_steps to instance attr at the end of all exit paths
        self._thinking_steps = thinking_steps
        return result

    async def send(self, processed_data: dict) -> dict:
        return processed_data


# ---------------------------------------------------------------------------
# Goal Pursuit Agent Loop
# ---------------------------------------------------------------------------

class GoalPursuitExecutor(HybridAgentExecutor):
    """
    Goal Pursuit Agent — hybrid agent loop with integrated planning, per-step
    evaluation, and autonomous replanning.

    Owns the full plan → execute step → evaluate → replan cycle in a single
    ``receive()`` call.  No external planner node required: creates its own
    plan, executes steps one by one using the hybrid tool loop, evaluates each
    outcome semantically via a dedicated LLM call, and revises the plan when a
    step fails — up to ``max_replan_depth`` times before giving up.

    Key differences from HybridAgentExecutor:
    - Creates its own step-by-step plan via LLM before executing.
    - Runs a focused hybrid loop per step (not one monolithic loop).
    - Evaluates each step's result with a separate LLM judge call.
    - On failure: replans remaining steps incorporating what went wrong.
    - Shares ``repl_state`` (variable storage) across all steps so large
      outputs from step N are accessible in step N+1.
    - Optionally synthesizes all step results into a final coherent answer.
    - Full episode persistence and streaming support.
    """

    _library_cache: dict = {"mtime": 0.0, "data": {}, "file_mtimes": {}}

    _DEFAULT_PLANNING_PROMPT = (
        "You are a task planner. Break the user's request into clear, executable steps.\n\n"
        "Rules:\n"
        "- Return a JSON array of steps.\n"
        "- Each step: {{\"step\": N, \"action\": \"verb phrase\", \"target\": \"what it applies to\", \"goal\": \"what success looks like\"}}\n"
        "- If the task is simple (single action or question), return a 1-step array.\n"
        "- Maximum {max_steps} steps. Keep steps atomic and independently testable.\n\n"
        "User request: {request}\n\n"
        "Respond with a JSON array only. No prose."
    )

    _DEFAULT_EVAL_PROMPT = (
        "You are evaluating whether an agent successfully completed a step.\n\n"
        "Step:\n"
        "  Action: {action}\n"
        "  Target: {target}\n"
        "  Success criterion: {goal}\n\n"
        "Agent response:\n{response}\n\n"
        "Did the agent successfully complete this step? "
        "Reply with JSON: {{\"success\": true/false, \"reason\": \"brief explanation\", "
        "\"extracted_result\": \"key output or finding from this step\"}}"
    )

    _DEFAULT_REPLAN_PROMPT = (
        "You are a task replanner. A step failed and you must create a revised plan.\n\n"
        "Original goal: {original_request}\n\n"
        "Completed steps:\n{completed_summary}\n\n"
        "Failed step:\n"
        "  Action: {failed_action}\n"
        "  Target: {failed_target}\n"
        "  Failure reason: {failure_reason}\n\n"
        "Create a revised plan for the REMAINING work (do not re-list completed steps). "
        "Work around the failure — skip, modify, or replace the failed step.\n\n"
        "Return a JSON array: [{{\"step\": N, \"action\": \"...\", \"target\": \"...\", "
        "\"goal\": \"what success looks like\"}}]\n\n"
        "Respond with JSON array only. No prose."
    )

    _DEFAULT_SYNTHESIS_PROMPT = (
        "You have completed a multi-step task. Synthesize the results into a final, "
        "coherent response for the user.\n\n"
        "Original goal: {original_request}\n\n"
        "Step results:\n{step_summary}\n\n"
        "Provide a complete, well-structured final answer."
    )

    # ------------------------------------------------------------------
    # Planning helpers
    # ------------------------------------------------------------------

    async def _parse_plan_response(self, content: str, max_steps: int, step_offset: int = 0) -> list:
        """Parse LLM JSON plan response into a list of step dicts."""
        import re
        content = re.sub(r'//.*', '', content)
        content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
        content = content.strip()
        try:
            parsed = json.loads(content)
            if isinstance(parsed, list):
                plan = []
                for i, step in enumerate(parsed[:max_steps]):
                    if isinstance(step, dict):
                        plan.append({
                            "step": step_offset + i + 1,
                            "action": step.get("action", step.get("task", "unknown")),
                            "target": step.get("target", step.get("query", "")),
                            "goal": step.get("goal", step.get("success_criterion", step.get("action", ""))),
                        })
                return plan
        except (json.JSONDecodeError, ValueError):
            pass
        return []

    async def _create_plan(self, user_request: str, config: dict) -> list:
        """Call LLM to create an initial step-by-step plan."""
        max_steps = int(config.get("max_steps", 10))
        prompt = config.get("planning_prompt", self._DEFAULT_PLANNING_PROMPT)
        prompt = prompt.replace("{request}", user_request).replace("{max_steps}", str(max_steps))
        model = config.get("model") or settings.get("default_model")
        response = await self._llm_with_retry(
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_request},
            ],
            model=model,
            temperature=float(config.get("planning_temperature", 0.1)),
            max_tokens=max(500, 60 * max_steps),
            tools=[],
            max_retries=int(config.get("max_llm_retries", 3)),
            retry_delay=float(config.get("retry_delay", 1.0)),
        )
        if not response or "choices" not in response:
            return []
        content = response["choices"][0]["message"].get("content", "")
        return await self._parse_plan_response(content, max_steps)

    async def _evaluate_step(self, step: dict, execution_result: str, config: dict) -> dict:
        """LLM judge: did the step's result actually satisfy its goal?

        Returns ``{"success": bool, "reason": str, "extracted_result": str}``.
        """
        import re
        prompt = config.get("eval_prompt", self._DEFAULT_EVAL_PROMPT)
        prompt = (
            prompt
            .replace("{action}", step.get("action", ""))
            .replace("{target}", step.get("target", ""))
            .replace("{goal}", step.get("goal", step.get("action", "")))
            .replace("{response}", execution_result[:3000])
        )
        model = config.get("model") or settings.get("default_model")
        response = await self._llm_with_retry(
            messages=[
                {"role": "system", "content": "You evaluate task completion. Reply with JSON only."},
                {"role": "user", "content": prompt},
            ],
            model=model,
            temperature=float(config.get("eval_temperature", 0.1)),
            max_tokens=300,
            tools=[],
            max_retries=int(config.get("max_llm_retries", 3)),
            retry_delay=float(config.get("retry_delay", 1.0)),
        )
        if not response or "choices" not in response:
            return {"success": False, "reason": "Evaluator LLM failed", "extracted_result": ""}

        raw = response["choices"][0]["message"].get("content", "").strip()
        # Strip markdown code fences before JSON extraction
        raw = re.sub(r'^```(?:json)?\s*', '', raw, flags=re.MULTILINE)
        raw = re.sub(r'^```\s*$', '', raw, flags=re.MULTILINE)
        raw = re.sub(r'//.*', '', raw)
        raw = re.sub(r'/\*.*?\*/', '', raw, flags=re.DOTALL)
        try:
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            if match:
                parsed = json.loads(match.group())
                return {
                    "success": bool(parsed.get("success", False)),
                    "reason": str(parsed.get("reason", "")),
                    "extracted_result": str(parsed.get("extracted_result", "")),
                }
        except (json.JSONDecodeError, ValueError):
            pass
        return {
            "success": False,
            "reason": f"Evaluator response could not be parsed: {raw[:200]}",
            "extracted_result": "",
        }

    async def _replan(
        self,
        original_request: str,
        completed_steps: list,
        step_results: list,
        failed_step: dict,
        failure_reason: str,
        config: dict,
    ) -> list:
        """Generate a revised plan for remaining work after a step failure."""
        max_steps = int(config.get("max_steps", 10))
        if completed_steps:
            lines = []
            for i, step in enumerate(completed_steps):
                r = step_results[i] if i < len(step_results) else {}
                extracted = r.get("extracted_result", "")[:150]
                suffix = f" → {extracted}" if extracted else ""
                lines.append(f"  ✓ {step.get('action', '?')}: {step.get('target', '')}{suffix}")
            completed_summary = "\n".join(lines)
        else:
            completed_summary = "(none)"

        prompt = config.get("replan_prompt", self._DEFAULT_REPLAN_PROMPT)
        prompt = (
            prompt
            .replace("{original_request}", original_request)
            .replace("{completed_summary}", completed_summary)
            .replace("{failed_action}", failed_step.get("action", ""))
            .replace("{failed_target}", failed_step.get("target", ""))
            .replace("{failure_reason}", failure_reason)
        )
        model = config.get("model") or settings.get("default_model")
        response = await self._llm_with_retry(
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"Create a revised plan to achieve: {original_request}"},
            ],
            model=model,
            temperature=float(config.get("planning_temperature", 0.1)),
            max_tokens=max(500, 60 * max_steps),
            tools=[],
            max_retries=int(config.get("max_llm_retries", 3)),
            retry_delay=float(config.get("retry_delay", 1.0)),
        )
        if not response or "choices" not in response:
            return []
        content = response["choices"][0]["message"].get("content", "")
        return await self._parse_plan_response(content, max_steps, step_offset=len(completed_steps))

    def _build_plan_context(
        self,
        completed_steps: list,
        remaining_steps: list,
        current_idx: int,
        step_results: list,
    ) -> str:
        """Format a progress-aware plan context string."""
        lines = []
        for i, step in enumerate(completed_steps):
            r = step_results[i] if i < len(step_results) else {}
            extracted = r.get("extracted_result", "")[:100]
            suffix = f" → {extracted}" if extracted else ""
            lines.append(
                f"✓ {step.get('step', i+1)}. {step.get('action','?')}: "
                f"{step.get('target','')}{suffix} (DONE)"
            )
        for i, step in enumerate(remaining_steps):
            prefix = "→" if i == current_idx else " "
            tag = " (CURRENT)" if i == current_idx else ""
            lines.append(
                f"{prefix} {step.get('step','?')}. {step.get('action','?')}: "
                f"{step.get('target','')}{tag}"
            )
        total = len(completed_steps) + len(remaining_steps)
        done = len(completed_steps)
        return f"## Execution Plan\nProgress: {done}/{total} steps done.\n" + "\n".join(lines)

    def _build_step_system_prompt(
        self,
        step: dict,
        plan_context: str,
        input_data: dict,
        config: dict,
        large_output_threshold: int,
        repl_state: dict | None = None,
    ) -> str:
        """Assemble the system prompt for a single step execution."""
        parts = [plan_context]
        if config.get("include_memory_context", True):
            v = input_data.get("_memory_context")
            if v:
                parts.append(f"## User Memories\n{v}")
        if config.get("include_knowledge_context", True):
            v = input_data.get("knowledge_context")
            if v:
                parts.append(f"## Relevant Knowledge\n{v}")
        if config.get("include_reasoning_context", True):
            v = input_data.get("reasoning_context")
            if v:
                parts.append(f"## Previous Reasoning\n{v}")
        parts.append(self._build_variable_system_note(large_output_threshold))
        if repl_state and repl_state.get("variables"):
            var_names = [k for k in repl_state["variables"] if k.startswith("step_")]
            if var_names:
                var_list = ", ".join(f'"{v}"' for v in var_names)
                parts.append(
                    f"## Available Step Results\n"
                    f"Previous steps stored their full outputs as variables: {var_list}.\n"
                    f"Use GetVariable(\"<name>\") to retrieve any of them."
                )
        parts.append(
            f"## Current Step\n"
            f"Action: {step.get('action', '')}\n"
            f"Target: {step.get('target', '')}\n"
            f"Success criterion: {step.get('goal', step.get('action', ''))}"
        )
        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def receive(self, input_data: dict, config: dict = None) -> dict:
        """
        Goal pursuit loop.

        1. Extract user goal from messages.
        2. Create a step-by-step plan via LLM (or restore from episode).
        3. For each step:
           a. Execute via the hybrid tool loop.
           b. Evaluate the result with an LLM judge (skippable via config).
           c. On success: record and advance.
           d. On failure: replan up to ``max_replan_depth`` times.
        4. Optionally synthesize all step results into a final answer.

        Config keys (in addition to all HybridAgentExecutor config keys):
            max_steps (int, default 10): max steps in initial plan.
            max_iterations_per_step (int, default 10): hybrid loop iterations per step.
            enable_step_evaluation (bool, default True): LLM judge per step outcome.
            enable_synthesis (bool, default True): final LLM synthesis pass.
            planning_temperature (float, default 0.1)
            eval_temperature (float, default 0.1)
            planning_prompt / eval_prompt / replan_prompt / synthesis_prompt (str)

        Output keys:
            content, messages, plan, completed_steps, step_results,
            remaining_steps, replan_count, goal_pursuit_trace,
            agent_loop_trace, iterations, episode_id (when persistence enabled)
        """
        if input_data is None:
            input_data = {}
        config = config or {}

        # --- Session init ---
        try:
            sm = get_session_manager()
            sm.load_or_create_session()
            sm.log_agent_event("agent_start", {
                "node": "goal_pursuit",
                "input_keys": list(input_data.keys()),
            })
        except Exception as e:
            logger.warning(f"[GoalPursuit] Session init failed: {e}")
            sm = None

        # --- Config ---
        max_iterations_per_step = int(config.get("max_iterations_per_step", 10))
        max_tokens = int(config.get("max_tokens", 4096))
        temperature = float(config.get("temperature", 0.7))
        max_llm_retries = int(config.get("max_llm_retries", 3))
        retry_delay = float(config.get("retry_delay", 1.0))
        tool_error_strategy = str(config.get("tool_error_strategy", "continue"))
        timeout = float(config.get("timeout", 300))
        max_replan_depth = int(config.get("max_replan_depth", 3))
        large_output_threshold = int(config.get("large_output_threshold", 3000))
        enable_step_evaluation = bool(config.get("enable_step_evaluation", True))
        enable_synthesis = bool(config.get("enable_synthesis", True))
        max_step_retries = int(config.get("max_step_retries", 2))
        step_timeout = float(config.get("step_timeout", 0))
        stream_queue = config.get("_stream_queue")
        model = config.get("model") or settings.get("default_model")

        # --- Episode persistence ---
        enable_episode = config.get("enable_episode_persistence", False)
        episode_id = config.get("episode_id")
        checkpoint_interval = max(1, int(config.get("checkpoint_interval", 1)))
        episode = None
        iteration_count = 0

        if enable_episode and sm:
            try:
                existing = sm.load_episode_state()
                should_resume = (
                    existing
                    and existing.phase not in [EpisodeState.PHASE_COMPLETED, EpisodeState.PHASE_FAILED]
                    and (episode_id is None or existing.episode_id == episode_id)
                )
                if should_resume:
                    episode = existing
                    input_data.setdefault("plan", existing.plan)
                    input_data.setdefault("completed_steps", existing.completed_steps)
                    input_data.setdefault("current_step", existing.current_step)
                    iteration_count = len(existing.checkpoints)
                    if existing.messages and config.get("resume_from_episode_messages", True):
                        input_data.setdefault("messages", existing.messages)
                    existing.update_phase(EpisodeState.PHASE_EXECUTING)
                    logger.info(f"[GoalPursuit] Resuming episode {existing.episode_id}")
                elif episode_id or config.get("auto_create_episode", True):
                    episode = sm.create_episode(
                        input_data={"initial_input": list(input_data.keys())},
                        budgets={"max_replan_depth": max_replan_depth, "timeout": timeout},
                        metadata={"episode_id": episode_id} if episode_id else {},
                    )
                    logger.info(f"[GoalPursuit] Created episode {episode.episode_id}")
            except Exception as e:
                logger.warning(f"[GoalPursuit] Episode init failed: {e}")

        # --- Replan depth guard ---
        replan_count = int(input_data.get("replan_count", 0))
        if episode:
            replan_count = max(replan_count, episode.replan_count)

        messages = input_data.get("messages", [])
        if not messages:
            result = input_data.copy()
            result["content"] = ""
            result["agent_loop_error"] = "No messages provided"
            return result

        # Extract user goal
        user_goal = ""
        for msg in reversed(messages):
            if isinstance(msg, dict) and msg.get("role") == "user":
                content = msg.get("content", "")
                user_goal = content if isinstance(content, str) else (
                    " ".join(
                        p.get("text", "") for p in content
                        if isinstance(p, dict) and p.get("type") == "text"
                    )
                )
                if user_goal:
                    break

        if not user_goal:
            result = input_data.copy()
            result["content"] = ""
            result["agent_loop_error"] = "No user message found"
            return result

        # --- Load tools and tool library ---
        tools_def = self._load_tools()
        tools_list = []
        for tool_name, tool_data in (tools_def or {}).items():
            if not isinstance(tool_data, dict) or not tool_data.get("enabled", True):
                continue
            if tool_name in self._EXCLUDED_TOOLS:
                continue
            definition = tool_data.get("definition")
            if definition:
                tools_list.append(definition)
        tool_library = self._load_tool_library()

        # Shared repl_state: variable storage is shared across all steps so
        # step N+1 can access large outputs stored by step N.
        repl_state: dict = {
            "variables": {},
            "_var_counts": {},
            "sub_call_count": 0,
            "max_sub_calls": int(config.get("max_sub_calls", 20)),
            "recursion_depth": 0,
            "max_recursion_depth": int(config.get("max_recursion_depth", 3)),
            "estimated_cost": 0.0,
            "max_cost_usd": float(config.get("max_cost_usd", 1.0)),
            "sub_call_model": config.get("sub_call_model") or model,
        }

        goal_pursuit_trace: list = []
        thinking_steps: list = []
        total_iterations = 0

        async def _execute():
            nonlocal replan_count, iteration_count, total_iterations

            # Step 1: create or restore plan
            remaining_steps = list(input_data.get("plan", []))
            completed_steps = list(input_data.get("completed_steps", []))
            step_results: list = list(input_data.get("step_results", []))

            if not remaining_steps:
                logger.info(f"[GoalPursuit] Creating plan for: {user_goal[:100]}")
                remaining_steps = await _create_plan_with_trace()
                if not remaining_steps:
                    return await _fallback_hybrid()

            # Step 2: execute steps
            step_idx = 0
            consecutive_failures = 0
            while step_idx < len(remaining_steps):
                if replan_count > max_replan_depth:
                    break

                step = remaining_steps[step_idx]
                step_trace = {
                    "step": step.get("step"),
                    "action": step.get("action"),
                    "target": step.get("target"),
                    "replan_count": replan_count,
                }

                plan_context = self._build_plan_context(
                    completed_steps, remaining_steps, step_idx, step_results
                )
                system_prompt = self._build_step_system_prompt(
                    step, plan_context, input_data, config, large_output_threshold, repl_state
                )
                step_messages = [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": (
                            f"Complete this step: {step.get('action', '')}: {step.get('target', '')}.\n"
                            f"Original goal: {user_goal}"
                        ),
                    },
                ]

                logger.info(
                    f"[GoalPursuit] Executing step {step.get('step')}: {step.get('action')}"
                )
                await _push_thinking({
                    "type": "tool_call",
                    "name": f"Step {step.get('step', step_idx + 1)}",
                    "content": f"{step.get('action', '')}: {step.get('target', '')}",
                })
                _step_coro = self._run_hybrid_loop(
                    llm_messages=step_messages,
                    tools_list=tools_list,
                    tool_library=tool_library,
                    repl_state=repl_state,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    max_iterations=max_iterations_per_step,
                    max_llm_retries=max_llm_retries,
                    retry_delay=retry_delay,
                    tool_error_strategy=tool_error_strategy,
                    large_output_threshold=large_output_threshold,
                    trace=goal_pursuit_trace,
                    stream_queue=stream_queue,
                    thinking_steps=thinking_steps,
                )
                _step_timed_out = False
                try:
                    if step_timeout > 0:
                        final_response, iters, had_tool_error = await asyncio.wait_for(
                            _step_coro, timeout=step_timeout
                        )
                    else:
                        final_response, iters, had_tool_error = await _step_coro
                except asyncio.TimeoutError:
                    _step_timed_out = True
                    final_response, iters, had_tool_error = None, 0, True
                    logger.warning(
                        f"[GoalPursuit] Step {step.get('step')} timed out after {step_timeout}s"
                    )
                total_iterations += iters
                iteration_count += iters

                step_content = ""
                if final_response and "choices" in final_response:
                    step_content = (
                        final_response["choices"][0]["message"].get("content") or ""
                    )

                step_trace["iterations"] = iters
                step_trace["had_tool_error"] = had_tool_error
                step_trace["timed_out"] = _step_timed_out
                step_trace["content_preview"] = step_content[:200]

                # Evaluate step outcome
                if enable_step_evaluation:
                    evaluation = await self._evaluate_step(step, step_content, config)
                else:
                    has_error_flag = (
                        final_response.get("error") is not None if final_response else False
                    )
                    success = bool(step_content) and not has_error_flag
                    evaluation = {
                        "success": success,
                        "reason": (
                            "tool_error" if had_tool_error
                            else ("no_content" if not step_content else "ok")
                        ),
                        "extracted_result": step_content[:500],
                    }

                step_trace["evaluation"] = evaluation
                logger.info(
                    f"[GoalPursuit] Step {step.get('step')} "
                    f"success={evaluation['success']}, reason={evaluation['reason'][:80]}"
                )
                await _push_thinking({
                    "type": "tool_result",
                    "name": f"Step {step.get('step', step_idx + 1)} eval",
                    "content": evaluation["reason"][:250],
                    "success": evaluation["success"],
                })

                if evaluation["success"]:
                    completed_steps.append(step)
                    step_results.append(evaluation)
                    consecutive_failures = 0
                    # Store full step output as a named variable so later steps
                    # (and the synthesizer) can retrieve it via GetVariable.
                    var_key = f"step_{step.get('step', step_idx + 1)}_result"
                    repl_state["variables"][var_key] = step_content
                    step_idx += 1

                    if episode is not None and sm is not None:
                        try:
                            sm.save_episode_state(
                                phase=EpisodeState.PHASE_EXECUTING,
                                replan_count=replan_count,
                                completed_steps=[s.get("step") for s in completed_steps],
                                current_step=step_idx,
                                plan=remaining_steps[step_idx:] if step_idx < len(remaining_steps) else [],
                                add_checkpoint=True,
                            )
                        except Exception as e:
                            logger.warning(f"[GoalPursuit] Episode checkpoint failed: {e}")
                else:
                    consecutive_failures += 1
                    failure_reason = evaluation.get("reason", "Unknown failure")
                    if _step_timed_out:
                        failure_reason = f"Step timed out after {step_timeout}s"

                    # Skip step if it has failed too many consecutive times,
                    # preserving the global replan budget for other steps.
                    if consecutive_failures >= max_step_retries:
                        step_trace["skipped"] = True
                        step_trace["skip_reason"] = (
                            f"Step failed {consecutive_failures} consecutive times; skipping"
                        )
                        logger.warning(
                            f"[GoalPursuit] Step {step.get('step')} skipped after "
                            f"{consecutive_failures} consecutive failures"
                        )
                        consecutive_failures = 0
                        step_idx += 1
                        goal_pursuit_trace.append(step_trace)
                        continue

                    if replan_count >= max_replan_depth:
                        step_trace["replan_skipped"] = "max_replan_depth reached"
                        goal_pursuit_trace.append(step_trace)
                        break

                    replan_count += 1
                    logger.info(
                        f"[GoalPursuit] Replanning (attempt {replan_count}/{max_replan_depth}): "
                        f"{failure_reason[:80]}"
                    )
                    await _push_thinking({
                        "type": "tool_call",
                        "name": f"Replan {replan_count}/{max_replan_depth}",
                        "content": f"Reason: {failure_reason[:200]}",
                    })
                    self._log_agent_event("goal_pursuit_replan", {
                        "replan_count": replan_count,
                        "failed_step": step.get("step"),
                        "failure_reason": failure_reason,
                    })

                    new_remaining = await self._replan(
                        original_request=user_goal,
                        completed_steps=completed_steps,
                        step_results=step_results,
                        failed_step=step,
                        failure_reason=failure_reason,
                        config=config,
                    )
                    step_trace["replanned"] = True
                    step_trace["new_steps_count"] = len(new_remaining)

                    if new_remaining:
                        remaining_steps = new_remaining
                        step_idx = 0
                        consecutive_failures = 0
                    else:
                        step_trace["replan_empty"] = True
                        consecutive_failures = 0
                        step_idx += 1

                    # Checkpoint partial progress after replan
                    if episode is not None and sm is not None:
                        try:
                            sm.save_episode_state(
                                phase=EpisodeState.PHASE_REPLANNING,
                                replan_count=replan_count,
                                completed_steps=[s.get("step") for s in completed_steps],
                                current_step=step_idx,
                                plan=remaining_steps[step_idx:] if step_idx < len(remaining_steps) else [],
                                add_checkpoint=False,
                            )
                        except Exception as e:
                            logger.warning(f"[GoalPursuit] Episode checkpoint (replan) failed: {e}")

                goal_pursuit_trace.append(step_trace)

            # Step 3: synthesize final answer
            if completed_steps and enable_synthesis and len(completed_steps) > 1:
                final_content = await _synthesize(completed_steps, step_results)
            elif completed_steps:
                final_content = step_results[-1].get("extracted_result", "") if step_results else ""
                if not final_content and goal_pursuit_trace:
                    final_content = goal_pursuit_trace[-1].get("content_preview", "")
            else:
                final_content = ""

            remaining_after = remaining_steps[step_idx:] if step_idx <= len(remaining_steps) else []
            result = input_data.copy()
            result["messages"] = messages
            result["content"] = final_content
            result["plan"] = remaining_steps
            result["completed_steps"] = completed_steps
            result["step_results"] = step_results
            result["remaining_steps"] = remaining_after
            result["replan_count"] = replan_count
            result["iterations"] = total_iterations
            result["goal_pursuit_trace"] = goal_pursuit_trace
            result["agent_loop_trace"] = goal_pursuit_trace
            result["repl_state"] = {
                "variables": list(repl_state["variables"].keys()),
                "variable_count": len(repl_state["variables"]),
            }
            result["replan_needed"] = bool(remaining_after)
            result["replan_depth_exceeded"] = replan_count >= max_replan_depth and bool(remaining_after)

            if episode is not None and sm is not None:
                try:
                    final_phase = (
                        EpisodeState.PHASE_COMPLETED
                        if not remaining_after and final_content
                        else EpisodeState.PHASE_FAILED
                    )
                    sm.save_episode_state(
                        phase=final_phase,
                        replan_count=replan_count,
                        completed_steps=[s.get("step") for s in completed_steps],
                        current_step=len(completed_steps),
                        plan=remaining_after,
                        add_checkpoint=True,
                    )
                    result["episode_id"] = episode.episode_id
                except Exception as e:
                    logger.warning(f"[GoalPursuit] Episode finalization failed: {e}")

            self._log_agent_event("goal_pursuit_end", {
                "completed_steps": len(completed_steps),
                "total_steps": len(remaining_steps),
                "replan_count": replan_count,
                "total_iterations": total_iterations,
            })
            self._thinking_steps = thinking_steps
            return result

        async def _push_thinking(entry: dict):
            """Append a thinking event and flush to stream_queue."""
            thinking_steps.append(entry)
            if stream_queue:
                await stream_queue.put({"type": "thinking", "content": list(thinking_steps)})

        async def _create_plan_with_trace() -> list:
            plan = await self._create_plan(user_goal, config)
            self._log_agent_event("goal_pursuit_plan_created", {
                "steps": len(plan),
                "goal_preview": user_goal[:100],
            })
            step_lines = "\n".join(
                f"  {i + 1}. {s.get('action', '?')}: {s.get('target', '')}"
                for i, s in enumerate(plan[:8])
            )
            await _push_thinking({
                "type": "tool_call",
                "name": "Plan",
                "content": f"Goal: {user_goal[:120]}\n{len(plan)} steps:\n{step_lines}",
            })
            return plan

        async def _synthesize(completed: list, results: list) -> str:
            lines = []
            for i, step in enumerate(completed):
                r = results[i] if i < len(results) else {}
                extracted = r.get("extracted_result", "")[:300]
                var_key = f"step_{step.get('step', i + 1)}_result"
                hint = (
                    f' (full output available via GetVariable("{var_key}"))'
                    if var_key in repl_state["variables"] else ""
                )
                lines.append(
                    f"Step {step.get('step')}: {step.get('action')}: {step.get('target')}\n"
                    f"Result summary: {extracted}{hint}"
                )
            step_summary = "\n\n".join(lines)
            prompt = config.get("synthesis_prompt", self._DEFAULT_SYNTHESIS_PROMPT)
            prompt = (
                prompt
                .replace("{original_request}", user_goal)
                .replace("{step_summary}", step_summary)
            )
            var_note = self._build_variable_system_note(large_output_threshold)
            synth_messages = [
                {"role": "system", "content": f"{prompt}\n\n{var_note}"},
                {"role": "user", "content": f"Provide a final answer for: {user_goal}"},
            ]
            final_response, _, _ = await self._run_hybrid_loop(
                llm_messages=synth_messages,
                tools_list=tools_list,
                tool_library=tool_library,
                repl_state=repl_state,
                model=model,
                temperature=float(config.get("temperature", 0.7)),
                max_tokens=max_tokens,
                max_iterations=5,
                max_llm_retries=int(config.get("max_llm_retries", 3)),
                retry_delay=float(config.get("retry_delay", 1.0)),
                tool_error_strategy=tool_error_strategy,
                large_output_threshold=large_output_threshold,
                trace=goal_pursuit_trace,
                stream_queue=stream_queue,
                thinking_steps=thinking_steps,
            )
            if final_response and "choices" in final_response:
                return final_response["choices"][0]["message"].get("content") or ""
            return ""

        async def _fallback_hybrid() -> dict:
            """Empty plan — run hybrid loop directly on original messages."""
            logger.info("[GoalPursuit] Empty plan, falling back to direct hybrid execution")
            llm_messages = list(messages)
            context_prompt = self._build_system_prompt(input_data, config)
            var_note = self._build_variable_system_note(large_output_threshold)
            full_prompt = "\n\n".join(filter(None, [context_prompt, var_note]))
            has_system = any(m.get("role") == "system" for m in llm_messages)
            if has_system:
                for m in llm_messages:
                    if m.get("role") == "system":
                        m["content"] = m["content"] + "\n\n" + full_prompt
                        break
            else:
                llm_messages.insert(0, {"role": "system", "content": full_prompt})

            final_response, iters, had_tool_error = await self._run_hybrid_loop(
                llm_messages=llm_messages,
                tools_list=tools_list,
                tool_library=tool_library,
                repl_state=repl_state,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                max_iterations=max_iterations_per_step,
                max_llm_retries=max_llm_retries,
                retry_delay=retry_delay,
                tool_error_strategy=tool_error_strategy,
                large_output_threshold=large_output_threshold,
                trace=goal_pursuit_trace,
                stream_queue=stream_queue,
                thinking_steps=thinking_steps,
            )
            content = ""
            if final_response and "choices" in final_response:
                content = final_response["choices"][0]["message"].get("content") or ""
            result = input_data.copy()
            result["messages"] = llm_messages
            result["content"] = content
            result["plan"] = []
            result["completed_steps"] = []
            result["step_results"] = []
            result["remaining_steps"] = []
            result["replan_count"] = 0
            result["iterations"] = iters
            result["goal_pursuit_trace"] = goal_pursuit_trace
            result["agent_loop_trace"] = goal_pursuit_trace
            self._thinking_steps = thinking_steps
            return result

        # --- Execute with timeout ---
        try:
            if timeout > 0:
                result = await asyncio.wait_for(_execute(), timeout=timeout)
            else:
                result = await _execute()
        except asyncio.TimeoutError:
            result = input_data.copy()
            result["content"] = ""
            result["agent_loop_error"] = f"Goal pursuit timed out after {timeout}s"
            result["iterations"] = total_iterations
            result["goal_pursuit_trace"] = goal_pursuit_trace
            result["agent_loop_trace"] = goal_pursuit_trace
            result["replan_count"] = replan_count
            self._thinking_steps = thinking_steps
            self._log_agent_event("goal_pursuit_timeout", {"timeout": timeout})
            return result
        except Exception as e:
            result = input_data.copy()
            result["content"] = ""
            result["agent_loop_error"] = str(e)
            result["iterations"] = total_iterations
            result["goal_pursuit_trace"] = goal_pursuit_trace
            result["agent_loop_trace"] = goal_pursuit_trace
            result["replan_count"] = replan_count
            self._thinking_steps = thinking_steps
            logger.exception(f"[GoalPursuit] Unexpected error: {e}")
            return result

        result["goal_pursuit_trace"] = goal_pursuit_trace
        result["agent_loop_trace"] = goal_pursuit_trace
        self._thinking_steps = thinking_steps
        return result

    async def send(self, processed_data: dict) -> dict:
        return processed_data
