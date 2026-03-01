"""
Tests for modules/agent_loop/node.py

Covers:
- Basic loop (no tool calls)
- Tool call execution and message threading
- Max iterations limit
- LLM retry on error / None response
- Exponential backoff timing
- All retries exhausted -> error dict returned
- Tool error with "continue" strategy
- Tool error with "stop" strategy
- Reflection-driven retry (not satisfied -> retry)
- Reflection-driven retry (satisfied -> stop immediately)
- Max reflection retries exceeded
- Timeout support
- Iteration tracing content
- System prompt injection from context fields
- send() passthrough
- get_executor_class() dispatcher
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, call, patch

from modules.agent_loop.node import AgentLoopExecutor, get_executor_class


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_llm_response(content: str = "Test response", tool_calls: list = None) -> dict:
    """Build a minimal valid LLM response dict."""
    message = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {"choices": [{"message": message}]}


def make_tool_call(
    name: str = "Calculator",
    args: str = '{"expression": "2+2"}',
    call_id: str = "call_1",
) -> dict:
    """Build a minimal tool_call dict as returned by the LLM."""
    return {"id": call_id, "function": {"name": name, "arguments": args}}


def make_executor(tools: dict = None, library: dict = None) -> AgentLoopExecutor:
    """
    Create an AgentLoopExecutor with _load_tools and _load_tool_library patched
    at the instance level so no context manager is needed.
    """
    executor = AgentLoopExecutor()
    executor._load_tools = MagicMock(return_value=tools or {})
    executor._load_tool_library = MagicMock(return_value=library or {})
    return executor


# ---------------------------------------------------------------------------
# Basic functionality
# ---------------------------------------------------------------------------

class TestAgentLoopBasic:
    """Tests for basic agent loop functionality."""

    @pytest.mark.asyncio
    async def test_none_input_returns_empty_dict(self):
        """None input should be treated as empty dict and returned unchanged (no messages)."""
        executor = make_executor()
        result = await executor.receive(None)
        assert result == {}

    @pytest.mark.asyncio
    async def test_empty_messages_returns_input_unchanged(self):
        """When messages list is empty the input should be returned as-is."""
        executor = make_executor()
        input_data = {"some_key": "some_value"}
        result = await executor.receive(input_data)
        assert result == input_data

    @pytest.mark.asyncio
    async def test_basic_no_tool_calls(self):
        """Single LLM call with no tool calls should complete in 1 iteration."""
        executor = make_executor()
        executor.llm.chat_completion = AsyncMock(return_value=make_llm_response("Hello!"))

        input_data = {"messages": [{"role": "user", "content": "Hi"}]}
        result = await executor.receive(input_data, config={"timeout": 0})

        assert result["content"] == "Hello!"
        assert result["iterations"] == 1
        assert "agent_loop_trace" in result
        assert len(result["agent_loop_trace"]) == 1
        assert result["agent_loop_trace"][0]["iteration"] == 1
        assert result["agent_loop_trace"][0]["tool_calls"] == []
        assert result["agent_loop_trace"][0]["errors"] == []

    @pytest.mark.asyncio
    async def test_tool_call_executed_and_result_appended(self):
        """Tool calls should be executed and results appended to messages."""
        tool_call = make_tool_call("Calculator", '{"expression": "2+2"}')
        responses = [
            make_llm_response("", tool_calls=[tool_call]),
            make_llm_response("The answer is 4"),
        ]

        executor = make_executor(library={"Calculator": "result = '4'"})
        executor.llm.chat_completion = AsyncMock(side_effect=responses)

        input_data = {"messages": [{"role": "user", "content": "What is 2+2?"}]}
        result = await executor.receive(input_data, config={"timeout": 0})

        assert result["content"] == "The answer is 4"
        assert result["iterations"] == 2
        assert len(result["agent_loop_trace"]) == 2
        assert "Calculator" in result["agent_loop_trace"][0]["tool_calls"]
        assert result["agent_loop_trace"][1]["tool_calls"] == []

    @pytest.mark.asyncio
    async def test_max_iterations_respected(self):
        """Loop must stop at max_iterations even if LLM keeps returning tool calls."""
        tool_call = make_tool_call("Calculator", '{"expression": "1+1"}')
        executor = make_executor(library={"Calculator": "result = '2'"})
        executor.llm.chat_completion = AsyncMock(
            return_value=make_llm_response("", tool_calls=[tool_call])
        )

        input_data = {"messages": [{"role": "user", "content": "Calculate"}]}
        result = await executor.receive(input_data, config={"max_iterations": 3, "timeout": 0})

        assert result["iterations"] == 3
        assert len(result["agent_loop_trace"]) == 3

    @pytest.mark.asyncio
    async def test_send_returns_data_unchanged(self):
        """send() must be a passthrough."""
        executor = make_executor()
        data = {"key": "value", "nested": {"a": 1}}
        result = await executor.send(data)
        assert result == data

    @pytest.mark.asyncio
    async def test_unknown_tool_produces_error_in_trace(self):
        """Calling a tool not in the library should produce an error in the trace."""
        tool_call = make_tool_call("NonExistentTool", "{}")
        responses = [
            make_llm_response("", tool_calls=[tool_call]),
            make_llm_response("I could not use the tool"),
        ]

        executor = make_executor(library={})
        executor.llm.chat_completion = AsyncMock(side_effect=responses)

        input_data = {"messages": [{"role": "user", "content": "Use a tool"}]}
        result = await executor.receive(input_data, config={"timeout": 0})

        assert any(
            "NonExistentTool" in err
            for err in result["agent_loop_trace"][0]["errors"]
        )

    @pytest.mark.asyncio
    async def test_output_contains_messages_key(self):
        """Result must always contain a 'messages' key when messages were provided."""
        executor = make_executor()
        executor.llm.chat_completion = AsyncMock(return_value=make_llm_response("Hi!"))

        input_data = {"messages": [{"role": "user", "content": "Hello"}]}
        result = await executor.receive(input_data, config={"timeout": 0})

        assert "messages" in result
        assert isinstance(result["messages"], list)

    @pytest.mark.asyncio
    async def test_response_key_present_in_output(self):
        """Result must contain the raw 'response' dict from the LLM."""
        llm_resp = make_llm_response("Raw response")
        executor = make_executor()
        executor.llm.chat_completion = AsyncMock(return_value=llm_resp)

        input_data = {"messages": [{"role": "user", "content": "Hi"}]}
        result = await executor.receive(input_data, config={"timeout": 0})

        assert result["response"] == llm_resp


# ---------------------------------------------------------------------------
# LLM Retry with Exponential Backoff
# ---------------------------------------------------------------------------

class TestLLMRetry:
    """Tests for _llm_with_retry() and its integration in receive()."""

    @pytest.mark.asyncio
    async def test_retry_on_error_response(self):
        """LLM should be retried when the response contains an 'error' key."""
        error_response = {"error": "Service unavailable"}
        success_response = make_llm_response("Success!")

        executor = make_executor()
        executor.llm.chat_completion = AsyncMock(
            side_effect=[error_response, success_response]
        )

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            result = await executor.receive(
                {"messages": [{"role": "user", "content": "Hi"}]},
                config={"max_llm_retries": 2, "retry_delay": 1.0, "timeout": 0},
            )

        assert result["content"] == "Success!"
        mock_sleep.assert_called_once_with(1.0)

    @pytest.mark.asyncio
    async def test_retry_on_none_response(self):
        """LLM should be retried when the response is None."""
        executor = make_executor()
        executor.llm.chat_completion = AsyncMock(
            side_effect=[None, make_llm_response("Recovered!")]
        )

        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await executor.receive(
                {"messages": [{"role": "user", "content": "Hi"}]},
                config={"max_llm_retries": 2, "retry_delay": 0.1, "timeout": 0},
            )

        assert result["content"] == "Recovered!"

    @pytest.mark.asyncio
    async def test_retry_on_missing_choices(self):
        """LLM should be retried when the response has no 'choices' key."""
        executor = make_executor()
        executor.llm.chat_completion = AsyncMock(
            side_effect=[{"model": "test"}, make_llm_response("OK")]
        )

        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await executor.receive(
                {"messages": [{"role": "user", "content": "Hi"}]},
                config={"max_llm_retries": 2, "retry_delay": 0.1, "timeout": 0},
            )

        assert result["content"] == "OK"

    @pytest.mark.asyncio
    async def test_exponential_backoff_delays(self):
        """Retry delays must follow the pattern: delay * 2^(attempt-1)."""
        executor = make_executor()
        executor.llm.chat_completion = AsyncMock(
            side_effect=[
                {"error": "Unavailable"},
                {"error": "Unavailable"},
                make_llm_response("OK"),
            ]
        )

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            await executor.receive(
                {"messages": [{"role": "user", "content": "Hi"}]},
                config={"max_llm_retries": 3, "retry_delay": 1.0, "timeout": 0},
            )

        sleep_calls = mock_sleep.call_args_list
        assert len(sleep_calls) == 2
        assert sleep_calls[0] == call(1.0)   # 1.0 * 2^0
        assert sleep_calls[1] == call(2.0)   # 1.0 * 2^1

    @pytest.mark.asyncio
    async def test_all_retries_exhausted_records_error_in_trace(self):
        """When all retries are exhausted the trace should record the LLM error."""
        executor = make_executor()
        executor.llm.chat_completion = AsyncMock(return_value={"error": "Persistent failure"})

        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await executor.receive(
                {"messages": [{"role": "user", "content": "Hi"}]},
                config={"max_llm_retries": 1, "retry_delay": 0.1, "timeout": 0},
            )

        assert len(result["agent_loop_trace"]) == 1
        assert len(result["agent_loop_trace"][0]["errors"]) > 0

    @pytest.mark.asyncio
    async def test_llm_exception_triggers_retry(self):
        """An exception raised by chat_completion should trigger a retry."""
        executor = make_executor()
        executor.llm.chat_completion = AsyncMock(
            side_effect=[Exception("Connection reset"), make_llm_response("Recovered!")]
        )

        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await executor.receive(
                {"messages": [{"role": "user", "content": "Hi"}]},
                config={"max_llm_retries": 2, "retry_delay": 0.1, "timeout": 0},
            )

        assert result["content"] == "Recovered!"

    @pytest.mark.asyncio
    async def test_no_retry_when_max_llm_retries_zero(self):
        """With max_llm_retries=0 there should be exactly one LLM call and no sleep."""
        executor = make_executor()
        executor.llm.chat_completion = AsyncMock(return_value={"error": "Fail"})

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            await executor.receive(
                {"messages": [{"role": "user", "content": "Hi"}]},
                config={"max_llm_retries": 0, "retry_delay": 1.0, "timeout": 0},
            )

        mock_sleep.assert_not_called()
        assert executor.llm.chat_completion.call_count == 1

    @pytest.mark.asyncio
    async def test_successful_first_attempt_no_sleep(self):
        """When the first LLM call succeeds, no sleep should occur."""
        executor = make_executor()
        executor.llm.chat_completion = AsyncMock(return_value=make_llm_response("OK"))

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            await executor.receive(
                {"messages": [{"role": "user", "content": "Hi"}]},
                config={"max_llm_retries": 3, "retry_delay": 1.0, "timeout": 0},
            )

        mock_sleep.assert_not_called()


# ---------------------------------------------------------------------------
# Tool Error Strategy
# ---------------------------------------------------------------------------

class TestToolErrorStrategy:
    """Tests for tool_error_strategy: continue vs stop."""

    @pytest.mark.asyncio
    async def test_continue_strategy_keeps_looping_after_tool_error(self):
        """With strategy='continue', a tool error should not stop the loop."""
        tool_call = make_tool_call("BrokenTool", "{}")
        responses = [
            make_llm_response("", tool_calls=[tool_call]),
            make_llm_response("Done despite tool error"),
        ]

        executor = make_executor(library={"BrokenTool": "raise Exception('boom')"})
        executor.llm.chat_completion = AsyncMock(side_effect=responses)

        result = await executor.receive(
            {"messages": [{"role": "user", "content": "Do something"}]},
            config={"tool_error_strategy": "continue", "timeout": 0},
        )

        assert result["content"] == "Done despite tool error"
        assert result["iterations"] == 2

    @pytest.mark.asyncio
    async def test_stop_strategy_halts_loop_on_tool_error(self):
        """With strategy='stop', a tool error should stop the loop immediately."""
        tool_call = make_tool_call("BrokenTool", "{}")
        responses = [
            make_llm_response("", tool_calls=[tool_call]),
            make_llm_response("This should not be reached"),
        ]

        executor = make_executor(library={"BrokenTool": "raise Exception('boom')"})
        executor.llm.chat_completion = AsyncMock(side_effect=responses)

        result = await executor.receive(
            {"messages": [{"role": "user", "content": "Do something"}]},
            config={"tool_error_strategy": "stop", "timeout": 0},
        )

        assert result["iterations"] == 1
        assert len(result["agent_loop_trace"][0]["errors"]) > 0

    @pytest.mark.asyncio
    async def test_tool_error_recorded_in_trace(self):
        """Tool errors should be recorded in the iteration trace."""
        tool_call = make_tool_call("BrokenTool", "{}")
        responses = [
            make_llm_response("", tool_calls=[tool_call]),
            make_llm_response("Finished"),
        ]

        executor = make_executor(library={"BrokenTool": "raise Exception('test error')"})
        executor.llm.chat_completion = AsyncMock(side_effect=responses)

        result = await executor.receive(
            {"messages": [{"role": "user", "content": "Go"}]},
            config={"tool_error_strategy": "continue", "timeout": 0},
        )

        first_trace = result["agent_loop_trace"][0]
        assert "BrokenTool" in first_trace["tool_calls"]
        assert any("BrokenTool" in e for e in first_trace["errors"])

    @pytest.mark.asyncio
    async def test_successful_tool_no_error_in_trace(self):
        """Successful tool execution should not produce errors in the trace."""
        tool_call = make_tool_call("Calculator", '{"expression": "1+1"}')
        responses = [
            make_llm_response("", tool_calls=[tool_call]),
            make_llm_response("Result is 2"),
        ]

        executor = make_executor(library={"Calculator": "result = '2'"})
        executor.llm.chat_completion = AsyncMock(side_effect=responses)

        result = await executor.receive(
            {"messages": [{"role": "user", "content": "Calculate"}]},
            config={"timeout": 0},
        )

        assert result["agent_loop_trace"][0]["errors"] == []

    @pytest.mark.asyncio
    async def test_multiple_tools_per_turn_all_recorded(self):
        """Multiple tool calls in one turn should all appear in the trace."""
        tool_calls = [
            make_tool_call("Calculator", '{"expression": "1+1"}', "call_1"),
            make_tool_call("Calculator", '{"expression": "2+2"}', "call_2"),
        ]
        responses = [
            make_llm_response("", tool_calls=tool_calls),
            make_llm_response("Done"),
        ]

        executor = make_executor(library={"Calculator": "result = '2'"})
        executor.llm.chat_completion = AsyncMock(side_effect=responses)

        result = await executor.receive(
            {"messages": [{"role": "user", "content": "Calculate twice"}]},
            config={"timeout": 0},
        )

        assert result["agent_loop_trace"][0]["tool_calls"].count("Calculator") == 2


# ---------------------------------------------------------------------------
# Reflection-Driven Retry
# ---------------------------------------------------------------------------

class TestReflectionDrivenRetry:
    """Tests for enable_reflection_retry and max_reflection_retries."""

    @pytest.mark.asyncio
    async def test_reflection_retry_when_not_satisfied(self):
        """When reflection is not satisfied, the loop should retry with improvement feedback."""
        responses = [
            make_llm_response("Bad answer"),
            make_llm_response("Better answer"),
        ]
        executor = make_executor()
        executor.llm.chat_completion = AsyncMock(side_effect=responses)

        reflection_results = [
            {
                "reflection": {"satisfied": False, "reason": "Too vague", "needs_improvement": "Be more specific"},
                "satisfied": False,
            },
            {
                "reflection": {"satisfied": True, "reason": "Good", "needs_improvement": None},
                "satisfied": True,
            },
        ]
        call_count = [0]

        async def mock_reflection(input_data, reflection_config=None):
            idx = call_count[0]
            call_count[0] += 1
            base = input_data.copy()
            base.update(reflection_results[min(idx, len(reflection_results) - 1)])
            return base

        executor._run_reflection = mock_reflection

        result = await executor.receive(
            {"messages": [{"role": "user", "content": "Explain something"}]},
            config={"enable_reflection_retry": True, "max_reflection_retries": 2, "timeout": 0},
        )

        assert executor.llm.chat_completion.call_count == 2
        assert result["content"] == "Better answer"
        assert result["satisfied"] is True

    @pytest.mark.asyncio
    async def test_reflection_stops_when_satisfied(self):
        """When reflection is satisfied on first check, no retry should occur."""
        executor = make_executor()
        executor.llm.chat_completion = AsyncMock(return_value=make_llm_response("Great answer"))

        async def mock_reflection(input_data, reflection_config=None):
            r = input_data.copy()
            r["reflection"] = {"satisfied": True, "reason": "Perfect", "needs_improvement": None}
            r["satisfied"] = True
            return r

        executor._run_reflection = mock_reflection

        result = await executor.receive(
            {"messages": [{"role": "user", "content": "Tell me something"}]},
            config={"enable_reflection_retry": True, "max_reflection_retries": 3, "timeout": 0},
        )

        assert executor.llm.chat_completion.call_count == 1
        assert result["satisfied"] is True

    @pytest.mark.asyncio
    async def test_max_reflection_retries_respected(self):
        """Reflection retries must not exceed max_reflection_retries."""
        executor = make_executor()
        executor.llm.chat_completion = AsyncMock(return_value=make_llm_response("Mediocre"))

        async def mock_reflection(input_data, reflection_config=None):
            r = input_data.copy()
            r["reflection"] = {"satisfied": False, "reason": "Still not good", "needs_improvement": "Try harder"}
            r["satisfied"] = False
            return r

        executor._run_reflection = mock_reflection

        await executor.receive(
            {"messages": [{"role": "user", "content": "Do something"}]},
            config={"enable_reflection_retry": True, "max_reflection_retries": 2, "timeout": 0},
        )

        # Initial run + 2 reflection retries = 3 total LLM calls
        assert executor.llm.chat_completion.call_count == 3

    @pytest.mark.asyncio
    async def test_reflection_disabled_by_default(self):
        """Reflection retry should not run when enable_reflection_retry is False (default)."""
        executor = make_executor()
        executor.llm.chat_completion = AsyncMock(return_value=make_llm_response("Answer"))
        executor._run_reflection = AsyncMock()

        await executor.receive(
            {"messages": [{"role": "user", "content": "Hi"}]},
            config={"timeout": 0},
        )

        executor._run_reflection.assert_not_called()

    @pytest.mark.asyncio
    async def test_reflection_result_recorded_in_trace(self):
        """Reflection result should be recorded in the last trace entry."""
        executor = make_executor()
        executor.llm.chat_completion = AsyncMock(return_value=make_llm_response("Answer"))

        async def mock_reflection(input_data, reflection_config=None):
            r = input_data.copy()
            r["reflection"] = {"satisfied": True, "reason": "Good enough", "needs_improvement": None}
            r["satisfied"] = True
            return r

        executor._run_reflection = mock_reflection

        result = await executor.receive(
            {"messages": [{"role": "user", "content": "Hi"}]},
            config={"enable_reflection_retry": True, "timeout": 0},
        )

        last_trace = result["agent_loop_trace"][-1]
        assert "reflection" in last_trace
        assert last_trace["reflection"]["satisfied"] is True

    @pytest.mark.asyncio
    async def test_reflection_no_improvement_hint_stops_retry(self):
        """If not satisfied but needs_improvement is None, retry should not occur."""
        executor = make_executor()
        executor.llm.chat_completion = AsyncMock(return_value=make_llm_response("Answer"))

        async def mock_reflection(input_data, reflection_config=None):
            r = input_data.copy()
            r["reflection"] = {"satisfied": False, "reason": "Not great", "needs_improvement": None}
            r["satisfied"] = False
            return r

        executor._run_reflection = mock_reflection

        result = await executor.receive(
            {"messages": [{"role": "user", "content": "Hi"}]},
            config={"enable_reflection_retry": True, "max_reflection_retries": 3, "timeout": 0},
        )

        # No improvement hint → no retry despite not satisfied
        assert executor.llm.chat_completion.call_count == 1

    @pytest.mark.asyncio
    async def test_improvement_message_injected_into_conversation(self):
        """The improvement feedback message should be injected as a user message on retry."""
        responses = [
            make_llm_response("First answer"),
            make_llm_response("Second answer"),
        ]
        executor = make_executor()
        executor.llm.chat_completion = AsyncMock(side_effect=responses)

        captured_second_call_messages = []

        original_run_loop = executor._run_agent_loop

        call_num = [0]

        async def capturing_run_loop(llm_messages, **kwargs):
            call_num[0] += 1
            if call_num[0] == 2:
                captured_second_call_messages.extend(llm_messages)
            return await original_run_loop(llm_messages=llm_messages, **kwargs)

        executor._run_agent_loop = capturing_run_loop

        reflection_results = [
            {"reflection": {"satisfied": False, "reason": "Bad", "needs_improvement": "Add details"}, "satisfied": False},
            {"reflection": {"satisfied": True, "reason": "OK", "needs_improvement": None}, "satisfied": True},
        ]
        call_count = [0]

        async def mock_reflection(input_data, reflection_config=None):
            idx = call_count[0]
            call_count[0] += 1
            r = input_data.copy()
            r.update(reflection_results[min(idx, len(reflection_results) - 1)])
            return r

        executor._run_reflection = mock_reflection

        await executor.receive(
            {"messages": [{"role": "user", "content": "Tell me something"}]},
            config={"enable_reflection_retry": True, "max_reflection_retries": 2, "timeout": 0},
        )

        # The second call should include the improvement feedback message
        user_messages = [m for m in captured_second_call_messages if m.get("role") == "user"]
        improvement_messages = [m for m in user_messages if "needs improvement" in m.get("content", "").lower()]
        assert len(improvement_messages) >= 1
        assert "Add details" in improvement_messages[0]["content"]


# ---------------------------------------------------------------------------
# Timeout Support
# ---------------------------------------------------------------------------

class TestTimeout:
    """Tests for timeout support via asyncio.wait_for."""

    @pytest.mark.asyncio
    async def test_timeout_returns_error_result(self):
        """When the loop times out, result should contain agent_loop_error."""
        executor = make_executor()

        async def slow_llm(*args, **kwargs):
            await asyncio.sleep(10)
            return make_llm_response("Too late")

        executor.llm.chat_completion = slow_llm

        result = await executor.receive(
            {"messages": [{"role": "user", "content": "Hi"}]},
            config={"timeout": 0.05},
        )

        assert "agent_loop_error" in result
        assert "timed out" in result["agent_loop_error"].lower()
        assert "agent_loop_trace" in result

    @pytest.mark.asyncio
    async def test_timeout_zero_disables_timeout(self):
        """timeout=0 should disable the timeout entirely."""
        executor = make_executor()
        executor.llm.chat_completion = AsyncMock(return_value=make_llm_response("Fast response"))

        result = await executor.receive(
            {"messages": [{"role": "user", "content": "Hi"}]},
            config={"timeout": 0},
        )

        assert "agent_loop_error" not in result
        assert result["content"] == "Fast response"

    @pytest.mark.asyncio
    async def test_timeout_preserves_input_data_keys(self):
        """On timeout, original input_data keys should be preserved in result."""
        executor = make_executor()

        async def slow_llm(*args, **kwargs):
            await asyncio.sleep(10)
            return make_llm_response("Too late")

        executor.llm.chat_completion = slow_llm

        result = await executor.receive(
            {
                "messages": [{"role": "user", "content": "Hi"}],
                "session_id": "abc123",
                "user_id": "user_1",
            },
            config={"timeout": 0.05},
        )

        assert result.get("session_id") == "abc123"
        assert result.get("user_id") == "user_1"

    @pytest.mark.asyncio
    async def test_timeout_error_message_includes_duration(self):
        """The timeout error message should mention the configured timeout value."""
        executor = make_executor()

        async def slow_llm(*args, **kwargs):
            await asyncio.sleep(10)
            return make_llm_response("Too late")

        executor.llm.chat_completion = slow_llm

        result = await executor.receive(
            {"messages": [{"role": "user", "content": "Hi"}]},
            config={"timeout": 0.05},
        )

        assert "0.05" in result["agent_loop_error"]


# ---------------------------------------------------------------------------
# System Prompt Injection
# ---------------------------------------------------------------------------

class TestSystemPromptInjection:
    """Tests for context-based system prompt injection."""

    @pytest.mark.asyncio
    async def test_plan_context_injected_into_system_prompt(self):
        """plan_context should be injected as a system message."""
        executor = make_executor()
        captured = []

        async def capture_llm(messages, **kwargs):
            captured.extend(messages)
            return make_llm_response("Done")

        executor.llm.chat_completion = capture_llm

        await executor.receive(
            {
                "messages": [{"role": "user", "content": "Execute plan"}],
                "plan_context": "## Execution Plan\n1. Step one\n2. Step two",
            },
            config={"include_plan_in_context": True, "timeout": 0},
        )

        system_msgs = [m for m in captured if m.get("role") == "system"]
        assert len(system_msgs) == 1
        assert "Execution Plan" in system_msgs[0]["content"]

    @pytest.mark.asyncio
    async def test_memory_context_injected(self):
        """_memory_context should be injected into the system message."""
        executor = make_executor()
        captured = []

        async def capture_llm(messages, **kwargs):
            captured.extend(messages)
            return make_llm_response("Done")

        executor.llm.chat_completion = capture_llm

        await executor.receive(
            {
                "messages": [{"role": "user", "content": "Hi"}],
                "_memory_context": "User likes Python",
            },
            config={"include_memory_context": True, "timeout": 0},
        )

        system_msgs = [m for m in captured if m.get("role") == "system"]
        assert len(system_msgs) == 1
        assert "User Memories" in system_msgs[0]["content"]
        assert "User likes Python" in system_msgs[0]["content"]

    @pytest.mark.asyncio
    async def test_existing_system_message_is_extended(self):
        """If a system message already exists, context should be appended to it."""
        executor = make_executor()
        captured = []

        async def capture_llm(messages, **kwargs):
            captured.extend(messages)
            return make_llm_response("Done")

        executor.llm.chat_completion = capture_llm

        await executor.receive(
            {
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Hi"},
                ],
                "plan_context": "## Plan\n1. Do it",
            },
            config={"include_plan_in_context": True, "timeout": 0},
        )

        system_msgs = [m for m in captured if m.get("role") == "system"]
        assert len(system_msgs) == 1
        assert "You are a helpful assistant." in system_msgs[0]["content"]
        assert "Plan" in system_msgs[0]["content"]

    @pytest.mark.asyncio
    async def test_no_context_no_system_message_added(self):
        """Without any context fields, no system message should be injected."""
        executor = make_executor()
        captured = []

        async def capture_llm(messages, **kwargs):
            captured.extend(messages)
            return make_llm_response("Done")

        executor.llm.chat_completion = capture_llm

        await executor.receive(
            {"messages": [{"role": "user", "content": "Hi"}]},
            config={"timeout": 0},
        )

        system_msgs = [m for m in captured if m.get("role") == "system"]
        assert len(system_msgs) == 0

    @pytest.mark.asyncio
    async def test_knowledge_context_injected(self):
        """knowledge_context should be injected into the system message."""
        executor = make_executor()
        captured = []

        async def capture_llm(messages, **kwargs):
            captured.extend(messages)
            return make_llm_response("Done")

        executor.llm.chat_completion = capture_llm

        await executor.receive(
            {
                "messages": [{"role": "user", "content": "Hi"}],
                "knowledge_context": "Python was created by Guido van Rossum.",
            },
            config={"include_knowledge_context": True, "timeout": 0},
        )

        system_msgs = [m for m in captured if m.get("role") == "system"]
        assert len(system_msgs) == 1
        assert "Relevant Knowledge" in system_msgs[0]["content"]
        assert "Guido van Rossum" in system_msgs[0]["content"]

    @pytest.mark.asyncio
    async def test_context_disabled_via_config(self):
        """Setting include_plan_in_context=False should suppress plan injection."""
        executor = make_executor()
        captured = []

        async def capture_llm(messages, **kwargs):
            captured.extend(messages)
            return make_llm_response("Done")

        executor.llm.chat_completion = capture_llm

        await executor.receive(
            {
                "messages": [{"role": "user", "content": "Hi"}],
                "plan_context": "## Plan\n1. Do it",
            },
            config={"include_plan_in_context": False, "timeout": 0},
        )

        system_msgs = [m for m in captured if m.get("role") == "system"]
        # No system message should be added since plan injection is disabled
        assert len(system_msgs) == 0


# ---------------------------------------------------------------------------
# get_executor_class dispatcher
# ---------------------------------------------------------------------------

class TestGetExecutorClass:
    """Tests for the get_executor_class module-level dispatcher."""

    @pytest.mark.asyncio
    async def test_returns_agent_loop_executor_for_correct_id(self):
        """get_executor_class('agent_loop') should return AgentLoopExecutor."""
        cls = await get_executor_class("agent_loop")
        assert cls is AgentLoopExecutor

    @pytest.mark.asyncio
    async def test_returns_none_for_unknown_id(self):
        """get_executor_class with an unknown id should return None."""
        cls = await get_executor_class("unknown_node_type")
        assert cls is None

    @pytest.mark.asyncio
    async def test_executor_class_is_instantiable(self):
        """The returned class should be instantiable without arguments."""
        cls = await get_executor_class("agent_loop")
        instance = cls()
        assert isinstance(instance, AgentLoopExecutor)


# ---------------------------------------------------------------------------
# _llm_with_retry unit tests (direct method tests)
# ---------------------------------------------------------------------------

class TestLLMWithRetryDirect:
    """Direct unit tests for the _llm_with_retry helper method."""

    @pytest.mark.asyncio
    async def test_returns_valid_response_immediately(self):
        """Should return on the first attempt when response is valid."""
        executor = make_executor()
        valid_response = make_llm_response("Hello")
        executor.llm.chat_completion = AsyncMock(return_value=valid_response)

        result = await executor._llm_with_retry(
            messages=[{"role": "user", "content": "Hi"}],
            model="test-model",
            temperature=0.7,
            max_tokens=100,
            tools=[],
            max_retries=3,
            retry_delay=0.1,
        )

        assert result == valid_response
        assert executor.llm.chat_completion.call_count == 1

    @pytest.mark.asyncio
    async def test_returns_error_dict_after_all_retries(self):
        """Should return an error dict when all retries are exhausted."""
        executor = make_executor()
        executor.llm.chat_completion = AsyncMock(return_value={"error": "Always fails"})

        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await executor._llm_with_retry(
                messages=[{"role": "user", "content": "Hi"}],
                model="test-model",
                temperature=0.7,
                max_tokens=100,
                tools=[],
                max_retries=2,
                retry_delay=0.1,
            )

        assert "error" in result
        assert "attempt(s)" in result["error"]
        # 1 initial + 2 retries = 3 total calls
        assert executor.llm.chat_completion.call_count == 3

    @pytest.mark.asyncio
    async def test_passes_tools_to_llm(self):
        """Tools list should be forwarded to chat_completion."""
        executor = make_executor()
        executor.llm.chat_completion = AsyncMock(return_value=make_llm_response("OK"))

        tools = [{"type": "function", "function": {"name": "Calculator"}}]

        await executor._llm_with_retry(
            messages=[{"role": "user", "content": "Hi"}],
            model="test-model",
            temperature=0.7,
            max_tokens=100,
            tools=tools,
            max_retries=0,
            retry_delay=0.0,
        )

        call_kwargs = executor.llm.chat_completion.call_args.kwargs
        assert call_kwargs["tools"] == tools

    @pytest.mark.asyncio
    async def test_empty_tools_passes_none(self):
        """Empty tools list should be passed as None to avoid API errors."""
        executor = make_executor()
        executor.llm.chat_completion = AsyncMock(return_value=make_llm_response("OK"))

        await executor._llm_with_retry(
            messages=[{"role": "user", "content": "Hi"}],
            model="test-model",
            temperature=0.7,
            max_tokens=100,
            tools=[],
            max_retries=0,
            retry_delay=0.0,
        )

        call_kwargs = executor.llm.chat_completion.call_args.kwargs
        assert call_kwargs["tools"] is None


# ---------------------------------------------------------------------------
# Integration-style tests
# ---------------------------------------------------------------------------

class TestIntegration:
    """Integration-style tests combining multiple features."""

    @pytest.mark.asyncio
    async def test_retry_then_tool_call_then_finish(self):
        """LLM fails once, retries, then makes a tool call, then finishes."""
        tool_call = make_tool_call("Calculator", '{"expression": "3*3"}')
        responses = [
            {"error": "Temporary failure"},       # First attempt fails
            make_llm_response("", tool_calls=[tool_call]),  # Retry succeeds with tool call
            make_llm_response("The answer is 9"),  # After tool result
        ]

        executor = make_executor(library={"Calculator": "result = '9'"})
        executor.llm.chat_completion = AsyncMock(side_effect=responses)

        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await executor.receive(
                {"messages": [{"role": "user", "content": "What is 3*3?"}]},
                config={"max_llm_retries": 2, "retry_delay": 0.1, "timeout": 0},
            )

        assert result["content"] == "The answer is 9"
        assert result["iterations"] == 2  # 2 successful LLM calls (retry + tool result)

    @pytest.mark.asyncio
    async def test_full_pipeline_with_context_and_tools(self):
        """Full pipeline: memory context + plan context + tool call + final answer."""
        tool_call = make_tool_call("Calculator", '{"expression": "10+5"}')
        responses = [
            make_llm_response("", tool_calls=[tool_call]),
            make_llm_response("The result is 15"),
        ]

        executor = make_executor(library={"Calculator": "result = '15'"})
        executor.llm.chat_completion = AsyncMock(side_effect=responses)

        captured = []
        original_llm = executor.llm.chat_completion

        async def capture_and_call(messages, **kwargs):
            captured.extend(messages)
            return await original_llm(messages, **kwargs)

        executor.llm.chat_completion = capture_and_call

        result = await executor.receive(
            {
                "messages": [{"role": "user", "content": "Calculate 10+5"}],
                "plan_context": "## Plan\n1. Use Calculator",
                "_memory_context": "User prefers concise answers",
            },
            config={
                "include_plan_in_context": True,
                "include_memory_context": True,
                "timeout": 0,
            },
        )

        assert result["content"] == "The result is 15"
        assert result["iterations"] == 2
        # System message should contain both plan and memory context
        system_msgs = [m for m in captured if m.get("role") == "system"]
        assert len(system_msgs) >= 1
        combined = system_msgs[0]["content"]
        assert "Plan" in combined
        assert "User Memories" in combined

    @pytest.mark.asyncio
    async def test_input_data_preserved_in_output(self):
        """All original input_data keys should be preserved in the output."""
        executor = make_executor()
        executor.llm.chat_completion = AsyncMock(return_value=make_llm_response("Done"))

        input_data = {
            "messages": [{"role": "user", "content": "Hi"}],
            "session_id": "sess_42",
            "user_name": "Alice",
            "custom_field": {"nested": True},
        }

        result = await executor.receive(input_data, config={"timeout": 0})

        assert result["session_id"] == "sess_42"
        assert result["user_name"] == "Alice"
        assert result["custom_field"] == {"nested": True}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
