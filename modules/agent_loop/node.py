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
        # Initialize session manager for tracing
        try:
            self._session_manager = get_session_manager()
        except Exception:
            self._session_manager = None

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
        
        # Add from end until we hit limit
        tokens_used = 0
        for msg in reversed(non_system_messages):
            msg_tokens = self._estimate_tokens(msg.get("content", ""))
            if tokens_used + msg_tokens <= available_tokens:
                result.append(msg)  # Append to end (chronologically later)
                tokens_used += msg_tokens
            else:
                break
        
        # Reverse to restore chronological order (oldest first)
        result.reverse()
        
        # Add system message at the beginning if present
        if system_msg:
            result.insert(0, system_msg)
        
        return result

    def _estimate_messages_tokens(self, messages: list) -> int:
        """Estimate total token count across all messages."""
        total = 0
        for m in messages:
            content = m.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    p.get("text", "") for p in content if p.get("type") == "text"
                )
            total += self._estimate_tokens(str(content) if content else "")
        return total

    async def _compact_messages(self, messages: list, keep_last: int) -> list:
        """LLM-summarize old agent messages to free up context window.

        Preserves leading system messages verbatim.  Summarizes the oldest
        non-system messages into a single system-level summary, then appends
        the most recent ``keep_last`` non-system messages verbatim.

        Returns the original list unchanged if compaction is unnecessary or
        the LLM summarization fails (fail-safe: never discard context silently).
        """
        # Separate the leading system prefix from the conversation turns
        system_prefix = []
        conversation = []
        for msg in messages:
            if msg.get("role") == "system" and not conversation:
                system_prefix.append(msg)
            else:
                conversation.append(msg)

        # Not enough conversation turns to bother compacting
        if len(conversation) <= keep_last + 2:
            return messages

        old_messages = conversation[:-keep_last]
        recent_messages = conversation[-keep_last:]

        # Build a readable transcript for the summarization prompt
        lines = []
        for m in old_messages:
            role = m.get("role", "?").upper()
            content = m.get("content", "")
            if isinstance(content, list):
                text_parts = [p.get("text", "") for p in content if p.get("type") == "text"]
                content = " ".join(text_parts) or "[non-text content]"
            lines.append(f"{role}: {str(content)[:600]}")

        summary_prompt = [
            {
                "role": "user",
                "content": (
                    "Summarize the following agent reasoning steps, tool calls, and results "
                    "concisely in 3-5 sentences. Preserve key facts, decisions, tool outcomes, "
                    "and any information needed to continue the task:\n\n"
                    + "\n".join(lines)
                ),
            }
        ]

        try:
            result = await self.llm.chat_completion(
                messages=summary_prompt,
                max_tokens=400,
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
            f"({len(old_messages)} old turns summarized)"
        )
        return compacted

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
        """Load tool implementations from library directory with file-level mtime-based caching."""
        try:
            if os.path.exists(LIBRARY_DIR):
                # Use file-level fingerprints for cache invalidation
                # This catches file content edits that may not update directory mtime
                current_files = {}
                for filename in os.listdir(LIBRARY_DIR):
                    if filename.endswith(".py"):
                        tool_name = filename[:-3]
                        code_path = os.path.join(LIBRARY_DIR, filename)
                        try:
                            # Store file mtime for each file as fingerprint
                            current_files[tool_name] = os.path.getmtime(code_path)
                        except OSError:
                            pass
                
                # Check if any file has changed by comparing fingerprints
                cache_data = self.__class__._library_cache.get("data", {})
                cache_mtimes = self.__class__._library_cache.get("file_mtimes", {})
                
                needs_reload = False
                # Check for new or modified files
                for tool_name, mtime in current_files.items():
                    if tool_name not in cache_mtimes or cache_mtimes[tool_name] != mtime:
                        needs_reload = True
                        break
                # Check for deleted files
                for tool_name in cache_mtimes:
                    if tool_name not in current_files:
                        needs_reload = True
                        break
                
                if needs_reload or not cache_data:
                    library = {}
                    for filename in os.listdir(LIBRARY_DIR):
                        if filename.endswith(".py"):
                            tool_name = filename[:-3]
                            code_path = os.path.join(LIBRARY_DIR, filename)
                            with open(code_path, "r") as f:
                                library[tool_name] = f.read()
                    self.__class__._library_cache["data"] = library
                    self.__class__._library_cache["file_mtimes"] = current_files
                return self.__class__._library_cache.get("data", {})
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

        # Log tool call event
        if self._session_manager:
            try:
                self._session_manager.log_tool_call(func_name, args)
            except Exception:
                pass  # Don't fail if logging fails

        start_time = time.time()
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

        # Calculate duration
        duration_ms = (time.time() - start_time) * 1000

        # Log tool result event
        if self._session_manager:
            try:
                self._session_manager.log_tool_result(
                    func_name,
                    output,
                    success=success,
                    duration_ms=duration_ms
                )
            except Exception:
                pass  # Don't fail if logging fails

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

        # Return None instead of error dict so caller can handle gracefully
        return None

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

            # Context compaction: if enabled and messages exceed threshold, summarize old turns
            if compact_threshold > 0:
                current_tokens = self._estimate_messages_tokens(llm_messages)
                if current_tokens > compact_threshold:
                    llm_messages = await self._compact_messages(llm_messages, compact_keep_last)

            # Log LLM call event
            if self._session_manager:
                try:
                    self._session_manager.log_llm_call(model, tokens=None)
                except Exception:
                    pass  # Don't fail if logging fails

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
            
            # --- Accumulate thinking steps in real-time ---
            if not hasattr(self, '_thinking_steps'):
                self._thinking_steps = []
            
            content = assistant_message.get("content") or ""
            tool_calls = assistant_message.get("tool_calls", [])
            if tool_calls:
                for tc in tool_calls:
                    fn = tc.get("function", {})
                    name = fn.get("name", "?")
                    args = fn.get("arguments", "{}")
                    try:
                        args_dict = json.loads(args)
                        args_display = ", ".join(f"{k}={repr(v)[:80]}" for k, v in args_dict.items())
                    except Exception:
                        args_display = args[:160]
                    self._thinking_steps.append({
                        "type": "tool_call",
                        "name": name,
                        "content": f"{name}({args_display})"
                    })
            elif content.strip() and iteration < max_iterations - 1:
                # Intermediate text response (not final, since iteration continues)
                # But wait, it hits break below if no tool calls!
                # If no tool calls, it breaks, so this ONLY triggers if there ARE tool calls 
                # or if some other logic continues. 
                pass

            if stream_queue and tool_calls:
                await stream_queue.put({"type": "thinking", "content": list(self._thinking_steps)})

            # Append assistant message to conversation
            llm_messages.append(assistant_message)

            if not tool_calls:
                # No more tool calls — agent has finished
                trace.append(iteration_trace)
                break

            # Execute all tool calls in this turn
            tool_error_occurred = False
            for tool_call in tool_calls:
                tool_name = tool_call.get("function", {}).get("name", "unknown")
                tool_args = tool_call.get("function", {}).get("arguments", "{}")
                
                # ... [Existing Deduplication Logic] ...
                try:
                    args_dict = json.loads(tool_args)
                    normalized_args = json.dumps(args_dict, sort_keys=True)
                except (json.JSONDecodeError, TypeError):
                    normalized_args = tool_args
                call_signature = f"{tool_name}:{normalized_args}"
                if call_signature in seen_calls:
                    iteration_trace["errors"].append(
                        f"Loop detected: '{tool_name}' called with same arguments repeatedly"
                    )
                    break
                seen_calls.add(call_signature)
                
                iteration_trace["tool_calls"].append(tool_name)

                tool_result = await self._execute_tool(tool_call, tool_library)

                # Append tool result to real-time thinking trace
                success = tool_result.get("success", False if not tool_result.get("content") else True)
                display = (tool_result.get("content") or "").strip()[:400]
                self._thinking_steps.append({
                    "type": "tool_result",
                    "name": tool_name,
                    "content": display,
                    "success": success
                })
                if stream_queue:
                    await stream_queue.put({"type": "thinking", "content": list(self._thinking_steps)})

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

        # --- Session initialization ---
        # Load or create session for tracing
        try:
            sm = get_session_manager()
            session_id = sm.load_or_create_session()
            # Log agent start event
            sm.log_agent_event("agent_start", {
                "input_keys": list(input_data.keys()),
                "config_keys": list(config.keys())
            })
        except Exception as e:
            logger.warning(f"Failed to initialize session manager: {e}")
            sm = None
            session_id = None

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
        compact_threshold = int(config.get("compact_threshold", 0))       # 0 = disabled
        compact_keep_last = int(config.get("compact_keep_last", 6))
        
        # --- Re-planning depth tracking ---
        replan_count = int(input_data.get("replan_count", 0))
        
        # --- Episode Persistence Support ---
        # Check if episode persistence is enabled
        enable_episode = config.get("enable_episode_persistence", False)
        episode_id = config.get("episode_id")  # Optional custom episode ID
        checkpoint_interval = config.get("checkpoint_interval", 1)  # Save checkpoint every N iterations
        
        # Issue 6: Guard against checkpoint_interval = 0 crash (division by zero)
        if checkpoint_interval <= 0:
            logger.warning("checkpoint_interval must be >= 1, defaulting to 1")
            checkpoint_interval = 1
        
        episode = None
        iteration_count = 0  # Track iterations for checkpoint saving
        
        if enable_episode and sm:
            try:
                # Try to load existing episode state for resume
                existing_episode = sm.load_episode_state()
                
                # Check if we should resume or create new episode
                should_resume = (
                    existing_episode and 
                    existing_episode.phase not in [
                        EpisodeState.PHASE_COMPLETED, 
                        EpisodeState.PHASE_FAILED
                    ] and
                    # Only resume if episode_id matches or no episode_id specified
                    (episode_id is None or existing_episode.episode_id == episode_id)
                )
                
                if should_resume:
                    # Resume from existing episode
                    episode = existing_episode
                    # Restore state from episode
                    replan_count = episode.replan_count
                    input_data.setdefault("plan", episode.plan)
                    input_data.setdefault("completed_steps", episode.completed_steps)
                    input_data.setdefault("current_step", episode.current_step)
                    # Restore iteration count for checkpoint tracking
                    iteration_count = len(episode.checkpoints)
                    # Optionally restore messages for resume
                    if episode.messages and config.get("resume_from_episode_messages", True):
                        input_data.setdefault("messages", episode.messages)
                    
                    # Update phase to executing
                    episode.update_phase(EpisodeState.PHASE_EXECUTING)
                    
                    logger.info(f"Resuming episode {episode.episode_id} from phase {episode.phase}")
                elif episode_id or config.get("auto_create_episode", True):
                    # Create new episode
                    budgets = {
                        "max_iterations": max_iterations,
                        "max_replan_depth": max_replan_depth,
                        "timeout": timeout,
                        "max_context_tokens": max_context_tokens,
                    }
                    # FIX: episode_id already passed in metadata above - no need to overwrite
                    episode = sm.create_episode(
                        input_data={"initial_input": list(input_data.keys())},
                        budgets=budgets,
                        metadata={"episode_id": episode_id} if episode_id else {}
                    )
                    logger.info(f"Created new episode {episode.episode_id}")
            except Exception as e:
                logger.warning(f"Failed to load episode state: {e}")
        
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
                trace=agent_loop_trace,
                stream_queue=config.get("_stream_queue") if config else None,
                compact_threshold=compact_threshold,
                compact_keep_last=compact_keep_last,
            )

            # --- Episode Checkpoint Saving ---
            # Save checkpoint after each iteration if enabled
            # FIX: Use local variables instead of result (which isn't defined yet)
            if episode is not None and sm is not None:
                iteration_count += iterations
                # Save checkpoint at configured intervals or on completion
                if iteration_count % checkpoint_interval == 0 or iterations == 0:
                    try:
                        # Use local variables since result isn't defined yet
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

            if final_response and "choices" in final_response:
                result["content"] = final_response["choices"][0]["message"].get("content", "")

            # Issue 5: Re-planning detection - use structured error flags instead of naive text matching
            # Check if the agent actually failed to produce a useful response
            # Don't trigger re-planning just because max_iterations was reached
            content = result.get("content", "")
            response_text = str(final_response) if final_response else ""
            
            # Use structured error detection instead of naive substring matching
            # This prevents false-positive replanning on benign text like "error handling best practices"
            has_error_flag = final_response.get("error") is not None if final_response else False
            has_tool_failure = had_tool_error
            has_no_content = not content
            
            # Only trigger replan if there's a structured error, tool failure, or no content
            actually_failed = has_tool_failure or has_no_content or has_error_flag
            
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

            # --- Episode Finalization ---
            # Save final episode state after execution completes
            if episode is not None and sm is not None:
                try:
                    # Determine final phase based on execution result
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
                    # Add episode info to result for reference
                    result["episode_id"] = episode.episode_id
                except Exception as e:
                    logger.warning(f"Failed to save final episode state: {e}")

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

        # --- Episode Finalization ---
        # Save final episode state after execution completes
        if episode is not None and sm is not None:
            try:
                # Determine final phase based on execution result
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
                # Add episode info to result for reference
                result["episode_id"] = episode.episode_id
            except Exception as e:
                logger.warning(f"Failed to save final episode state: {e}")

        # Attach trace to final result
        result["agent_loop_trace"] = agent_loop_trace
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
    return None


class REPLEnvironmentExecutor:
    """
    REPL Environment Node - Initializes a persistent REPL state.
    
    This is the foundation of RLM processing. It:
    1. Extracts user input from messages
    2. Stores it as prompt_var in repl_state (NOT in context window)
    3. Builds metadata-only system prompt
    4. Initializes full repl_state structure including recursion tracking
    
    Config:
        - max_recursion_depth (int, default 3): Max sub-call depth
        - max_cost_usd (float, default 1.0): Hard cost stop
        - max_sub_calls (int, default 50): Max total sub-calls
        - sub_call_model (str): Model for sub-calls (defaults to root model)
    """
    
    async def receive(self, input_data: dict, config: dict = None) -> dict:
        if input_data is None:
            return None
            
        config = config or {}
        messages = input_data.get("messages", [])
        
        # Extract user content
        user_content = self._extract_user_content(messages)
        
        # Build comprehensive repl_state
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
        
        # Build metadata-only system prompt
        preview = user_content[:200] + "..." if len(user_content) > 200 else user_content
        system_content = self._build_repl_system_prompt({
            "prompt_length": len(user_content),
            "prompt_preview": preview,
            "prompt_type": self._classify_content(user_content),
            "max_recursion_depth": repl_state["max_recursion_depth"],
            "max_sub_calls": repl_state["max_sub_calls"],
        })
        
        result = input_data.copy()
        result["repl_state"] = repl_state
        result["_repl_initialized"] = True
        
        # Replace messages with metadata-only context
        result["messages"] = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": "Process the input using the REPL environment."}
        ]
        
        return result
    
    def _extract_user_content(self, messages: list) -> str:
        """Extract the actual user content from messages."""
        for msg in messages:
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if content:
                    return content
        return " ".join(m.get("content", "") for m in messages if m.get("role") == "user")

    def _classify_content(self, content: str) -> str:
        """Classify the type of content for metadata."""
        content_lower = content.lower()
        if "code" in content_lower or "def " in content_lower:
            return "code"
        elif len(content) > 100000:
            return "large_document"
        elif len(content) > 10000:
            return "long_text"
        return "standard"

    def _build_repl_system_prompt(self, metadata: dict) -> str:
        """Build the system prompt for REPL mode - metadata only, no actual content."""
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


class RLMAgentLoopExecutor:
    """
    RLM (Recursive Language Model) variant of the agent loop.
    
    Key differences from standard AgentLoopExecutor:
    1. REPL state is maintained across iterations - input stored as variable, not in context
    2. Tool outputs are stored as variables in repl_state, NOT injected into messages
    3. Only metadata about stdout goes back into LLM history (constant size)
    4. Terminates when set_final() is called
    
    This architecture enables processing arbitrarily long inputs (10M+ tokens) by:
    - Storing the actual content in repl_state, not in messages
    - Only passing constant-size metadata to the LLM
    - Allowing the LLM to write code to examine/decompose content via tools
    """
    
    # Class-level cache for tools
    _tools_cache = {"mtime": 0.0, "data": {}}
    _library_cache = {"mtime": 0.0, "data": {}}

    def __init__(self):
        self.llm = LLMBridge(
            base_url=settings.get("llm_api_url"),
            api_key=settings.get("llm_api_key")
        )
        from modules.tools.sandbox import ToolSandbox
        self._sandbox = ToolSandbox(timeout=30.0)
        # Initialize session manager for tracing
        try:
            self._session_manager = get_session_manager()
        except Exception:
            self._session_manager = None

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count: ~4 characters per token."""
        if not text:
            return 0
        return len(text) // 4

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
        """Load tool implementations from RLM library directory with mtime-based caching."""
        try:
            # Load from RLM library directory (RLM-specific tools: Peek, Search, Chunk, etc.)
            if os.path.exists(RLM_LIBRARY_DIR):
                dir_mtime = os.path.getmtime(RLM_LIBRARY_DIR)
                # Check if cache needs refresh (compare with RLM library mtime)
                cache_key = f"{RLM_LIBRARY_DIR}|{LIBRARY_DIR}"
                if dir_mtime > self.__class__._library_cache.get("mtime", 0):
                    library = {}
                    # First load RLM-specific tools
                    for filename in os.listdir(RLM_LIBRARY_DIR):
                        if filename.endswith(".py"):
                            tool_name = filename[:-3]
                            code_path = os.path.join(RLM_LIBRARY_DIR, filename)
                            with open(code_path, "r") as f:
                                library[tool_name] = f.read()
                    # Then load common tools from library (like SubCall)
                    if os.path.exists(LIBRARY_DIR):
                        for filename in os.listdir(LIBRARY_DIR):
                            if filename.endswith(".py"):
                                tool_name = filename[:-3]
                                # Don't override RLM-specific tools
                                if tool_name not in library:
                                    code_path = os.path.join(LIBRARY_DIR, filename)
                                    with open(code_path, "r") as f:
                                        library[tool_name] = f.read()
                    self.__class__._library_cache["data"] = library
                    self.__class__._library_cache["mtime"] = dir_mtime
                return self.__class__._library_cache.get("data", {})
        except OSError as e:
            logger.warning(f"Failed to load RLM tool library: {e}")
        return {}
    
    def _resolve_tool_name(self, func_name: str, tool_library: dict) -> str:
        """
        Resolve tool name with case-insensitive matching.
        Returns the actual tool name from the library if found.
        """
        # Direct match
        if func_name in tool_library:
            return func_name
        
        # Case-insensitive match
        func_lower = func_name.lower()
        for tool_name in tool_library:
            if tool_name.lower() == func_lower:
                return tool_name
        
        # Snake case to CamelCase conversion (e.g., set_variable -> SetVariable)
        parts = func_name.split('_')
        camel_case = ''.join(p.capitalize() for p in parts)
        if camel_case in tool_library:
            return camel_case
        
        # Reverse: CamelCase to snake_case
        snake_case = ''
        for i, char in enumerate(func_name):
            if char.isupper() and i > 0:
                snake_case += '_'
            snake_case += char.lower()
        if snake_case in tool_library:
            return snake_case
        
        return func_name  # Return original if not found

    def _extract_user_content(self, messages: list) -> str:
        """Extract the actual user content from messages."""
        for msg in messages:
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if content:
                    return content
        # Fallback: concatenate all user messages
        return " ".join(m.get("content", "") for m in messages if m.get("role") == "user")

    def _classify_content(self, content: str) -> str:
        """Classify the type of content for metadata."""
        content_lower = content.lower()
        if "code" in content_lower or "def " in content_lower or "function" in content_lower:
            return "code"
        elif len(content) > 100000:
            return "large_document"
        elif len(content) > 10000:
            return "long_text"
        else:
            return "standard"

    def _build_repl_system_prompt(self, metadata: dict) -> str:
        """Build the system prompt for REPL mode - metadata only, no actual content."""
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
        """
        Initialize REPL state from messages.
        
        Returns (repl_state, metadata, new_messages)
        """
        user_content = self._extract_user_content(messages)
        
        # Store content as environment variable, NOT in context
        repl_state = {
            "prompt_var": user_content,
            "variables": {},
            "stdout_history": [],
            "final": None,
            "iteration": 0
        }
        
        # Metadata that the LLM actually sees
        preview = user_content[:200] + "..." if len(user_content) > 200 else user_content
        metadata = {
            "prompt_length": len(user_content),
            "prompt_preview": preview,
            "prompt_type": self._classify_content(user_content),
        }
        
        # Build new messages with REPL system prompt only
        system_content = self._build_repl_system_prompt(metadata)
        new_messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": "Process the input using the REPL environment. Use tools to examine the content incrementally and call set_final() when done."}
        ]
        
        return repl_state, metadata, new_messages

    async def _execute_tool(self, tool_call: dict, tool_library: dict, repl_state: dict) -> dict:
        """Execute a tool call - sub_call is handled specially as a built-in."""
        func_name = tool_call["function"]["name"]
        try:
            args = json.loads(tool_call["function"]["arguments"])
        except (json.JSONDecodeError, KeyError):
            args = {}

        success = False
        
        # CRITICAL: sub_call is a built-in, NOT a library tool
        # It cannot be executed via exec() because it requires async LLM calls
        # Using run_until_complete inside an already-running event loop causes deadlock
        if func_name.lower() == "subcall" or func_name.lower() == "sub_call":
            return await self._execute_sub_call(args, repl_state)
        
        # Resolve tool name with case-insensitive matching
        actual_tool_name = self._resolve_tool_name(func_name, tool_library)
        
        # For all other tools, use sandbox execution
        args["_repl_state"] = repl_state
        
        if actual_tool_name in tool_library:
            code = tool_library[actual_tool_name]
            try:
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(
                    None,
                    self._sandbox.execute,
                    code,
                    {"args": args}
                )
                output = result.get("result", "Success (no result returned)")
                success = True
                
                # Update repl_state if tool returned state update
                if isinstance(result.get("_repl_state_update"), dict):
                    repl_state.update(result["_repl_state_update"])
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
    
    async def _execute_sub_call(self, args: dict, repl_state: dict) -> dict:
        """
        Execute sub_call as a built-in - NOT via exec().
        
        This is the KEY RLM feature that enables recursive processing.
        Must be async to avoid deadlock from run_until_complete in running loop.
        """
        # Check guardrails first
        if repl_state.get("sub_call_count", 0) >= repl_state.get("max_sub_calls", 50):
            return {
                "tool_call_id": "",
                "role": "tool",
                "name": "sub_call",
                "content": "Error: max_sub_calls limit reached",
                "success": False
            }
        
        if repl_state.get("recursion_depth", 0) >= repl_state.get("max_recursion_depth", 3):
            return {
                "tool_call_id": "",
                "role": "tool",
                "name": "sub_call",
                "content": "Error: max_recursion_depth limit reached",
                "success": False
            }
        
        # Check cost limit
        if repl_state.get("estimated_cost", 0) >= repl_state.get("max_cost_usd", 1.0):
            return {
                "tool_call_id": "",
                "role": "tool",
                "name": "sub_call",
                "content": "Error: max_cost_usd limit reached",
                "success": False
            }
        
        prompt = args.get("prompt", "")
        model = args.get("model") or repl_state.get("sub_call_model") or settings.get("default_model")
        max_tokens = args.get("max_tokens", 2000)
        
        if not prompt:
            return {
                "tool_call_id": "",
                "role": "tool",
                "name": "sub_call",
                "content": "Error: No prompt provided for sub_call",
                "success": False
            }
        
        try:
            # Increment counters BEFORE the call
            repl_state["sub_call_count"] = repl_state.get("sub_call_count", 0) + 1
            old_depth = repl_state.get("recursion_depth", 0)
            repl_state["recursion_depth"] = old_depth + 1
            
            # Make the async LLM call properly (not via run_until_complete)
            response = await self.llm.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                model=model,
                max_tokens=max_tokens
            )
            
            # Restore depth after call
            repl_state["recursion_depth"] = old_depth
            
            if response and "choices" in response:
                content = response["choices"][0]["message"].get("content", "")
                
                # Estimate cost (rough: ~4 chars per token, $0.001 per 1K tokens)
                estimated_tokens = len(prompt) // 4 + max_tokens
                cost = (estimated_tokens / 1000) * 0.001
                repl_state["estimated_cost"] = repl_state.get("estimated_cost", 0) + cost
                
                return {
                    "tool_call_id": "",
                    "role": "tool",
                    "name": "sub_call",
                    "content": content,
                    "success": True,
                    "model_used": model,
                    "sub_call_count": repl_state["sub_call_count"],
                    "estimated_cost": repl_state["estimated_cost"]
                }
            else:
                return {
                    "tool_call_id": "",
                    "role": "tool",
                    "name": "sub_call",
                    "content": f"Error in sub_call: {response.get('error', 'Unknown error') if response else 'No response'}",
                    "success": False
                }
        except Exception as e:
            repl_state["recursion_depth"] = old_depth  # Restore on error
            return {
                "tool_call_id": "",
                "role": "tool",
                "name": "sub_call",
                "content": f"Error in sub_call: {str(e)}",
                "success": False
            }

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
        """Call LLM with exponential backoff retry."""
        last_error = "Unknown error"

        for attempt in range(max_retries + 1):
            if attempt > 0:
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

                if response and "choices" in response and not response.get("error"):
                    return response

                if response:
                    last_error = response.get("error", "Response missing 'choices' key")
                else:
                    last_error = "LLM returned None"

            except Exception as e:
                last_error = str(e)

        # Return None instead of error dict so caller can handle gracefully
        return None

    async def _run_rlm_loop(
        self,
        initial_messages: list,
        repl_state: dict,
        tools_list: list,
        tool_library: dict,
        model: str,
        config: dict,
        trace: list
    ) -> tuple:
        """
        Core RLM loop: LLM generates code → executes tools → only metadata goes back.
        
        Key difference from standard loop:
        - Tool outputs stored in repl_state["variables"], NOT in messages
        - Only constant-size metadata appended to messages
        - Terminates when repl_state["final"] is set
        """
        max_iterations = config.get("max_iterations", 20)
        stdout_preview_length = config.get("stdout_preview_length", 500)
        max_llm_retries = config.get("max_llm_retries", 3)
        retry_delay = config.get("retry_delay", 1.0)
        temperature = config.get("temperature", 0.1)  # Low temp for code generation
        max_tokens = config.get("max_tokens", 2000)

        # Build initial LLM messages
        llm_messages = initial_messages.copy()

        for iteration in range(max_iterations):
            repl_state["iteration"] = iteration
            iteration_trace = {
                "iteration": iteration,
                "tool_calls": [],
                "errors": []
            }

            # Inject current REPL state as metadata (small, constant-size)
            state_metadata = {
                "iteration": iteration,
                "variables_set": list(repl_state["variables"].keys()),
                "stdout_entries": len(repl_state["stdout_history"]),
                "last_stdout_preview": (
                    repl_state["stdout_history"][-1][:stdout_preview_length]
                    if repl_state["stdout_history"] else "none"
                ),
                "final_set": repl_state["final"] is not None
            }

            # Add state metadata to messages (NOT the actual content)
            messages_with_state = llm_messages + [{
                "role": "system",
                "content": f"REPL state: {json.dumps(state_metadata)}"
            }]

            # LLM generates code
            response = await self._llm_with_retry(
                messages=messages_with_state,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=tools_list,
                max_retries=max_llm_retries,
                retry_delay=retry_delay
            )

            if not response or "choices" not in response:
                error_msg = response.get("error", "LLM returned no choices") if response else "LLM returned None"
                iteration_trace["errors"].append(error_msg)
                trace.append(iteration_trace)
                break

            assistant_message = response["choices"][0]["message"]
            tool_calls = assistant_message.get("tool_calls", [])

            if not tool_calls:
                # LLM returned text instead of code — capture as stdout
                content = assistant_message.get("content", "")
                if content:
                    repl_state["stdout_history"].append(content)
                    # Only metadata goes back, NOT the full content
                    llm_messages.append({
                        "role": "assistant", 
                        "content": f"[stdout: {len(content)} chars, preview: {content[:100]}...]"
                    })
                # Check if final was set in the text response
                if repl_state.get("final") is not None:
                    break
                break

            # Execute tool calls, pass repl_state through
            for tool_call in tool_calls:
                tool_name = tool_call.get("function", {}).get("name", "unknown")
                iteration_trace["tool_calls"].append(tool_name)

                tool_result = await self._execute_tool(tool_call, tool_library, repl_state)

                output_content = tool_result["content"]
                repl_state["stdout_history"].append(str(output_content))

                # CRITICAL: Only metadata goes into LLM history
                # This is what prevents context rot!
                stdout_metadata = (
                    f"[stdout: {len(str(output_content))} chars, "
                    f"preview: {str(output_content)[:stdout_preview_length]}...]"
                )
                llm_messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.get("id", ""),
                    "content": stdout_metadata  # NOT the full output
                })

            # Check if final answer has been set
            if repl_state.get("final") is not None:
                trace.append(iteration_trace)
                break
                
            trace.append(iteration_trace)

        return repl_state.get("final"), repl_state

    async def receive(self, input_data: dict, config: dict = None) -> dict:
        """
        RLM Agent Loop: processes arbitrarily long inputs via REPL environment.
        
        Input:
            - messages: conversation history (must include at least one user message)
            
        Config:
            - max_iterations (int, default 20): Maximum RLM iterations
            - max_tokens (int, default 2000): Max tokens per LLM call
            - temperature (float, default 0.1): Low temp for code generation
            - max_llm_retries (int, default 3): Retries on LLM failure
            - retry_delay (float, default 1.0): Base delay for backoff
            - stdout_preview_length (int, default 500): Preview size in metadata
            - sub_call_model (str, optional): Model for sub_call tool
            - model (str, optional): Main model (defaults to settings.get("default_model"))

        Output:
            - content: The final answer from set_final()
            - repl_state: Final REPL state for debugging
            - iterations: Number of iterations executed
            - rlm_trace: Per-iteration details
        """
        if input_data is None:
            input_data = {}

        config = config or {}

        # Configuration
        max_iterations = int(config.get("max_iterations", 20))
        max_tokens = int(config.get("max_tokens", 2000))
        temperature = float(config.get("temperature", 0.1))
        max_llm_retries = int(config.get("max_llm_retries", 3))
        retry_delay = float(config.get("retry_delay", 1.0))
        stdout_preview_length = int(config.get("stdout_preview_length", 500))
        model = config.get("model") or settings.get("default_model")

        # Guard: no messages → return input unchanged
        messages = input_data.get("messages", [])
        if not messages:
            result = input_data.copy()
            result["content"] = ""
            result["rlm_error"] = "No messages provided"
            return result

        # Step 1: Initialize REPL state
        # This extracts the user content and stores it in repl_state
        # instead of putting it in the LLM context
        repl_state, metadata, new_messages = self._init_repl_state(messages)

        # Step 2: Load tools
        tools_def = self._load_tools()
        tools_list = []
        if tools_def:
            for tool_name, tool_data in tools_def.items():
                if isinstance(tool_data, dict) and tool_data.get("enabled", True):
                    definition = tool_data.get("definition")
                    if definition:
                        tools_list.append(definition)

        tool_library = self._load_tool_library()

        # Step 3: Run RLM loop
        trace = []
        final_result, final_state = await self._run_rlm_loop(
            initial_messages=new_messages,
            repl_state=repl_state,
            tools_list=tools_list,
            tool_library=tool_library,
            model=model,
            config=config,
            trace=trace
        )

        # Step 4: Build result
        result = input_data.copy()
        result["content"] = final_result or ""
        result["repl_state"] = {
            "variables": final_state.get("variables", {}),
            "iteration": final_state.get("iteration", 0),
            "stdout_count": len(final_state.get("stdout_history", []))
        }
        result["iterations"] = final_state.get("iteration", 0) + 1
        result["rlm_trace"] = trace
        result["messages"] = new_messages  # Return the REPL messages (not full history)
        
        if not final_result:
            result["rlm_error"] = "No final answer set - max iterations reached or error"

        return result

    async def send(self, processed_data: dict) -> dict:
        return processed_data

