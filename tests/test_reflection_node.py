"""
Tests for modules/reflection/node.py

Covers:
- Satisfied response — reflection dict set, satisfied=True, messages unchanged
- Not satisfied — improvement message injected into messages
- Not satisfied — satisfied=False at top level for ConditionalRouter routing
- No messages / no content — defaults to satisfied=True (no blocking)
- LLM exception — defaults to satisfied=True (fail-safe)
- inject_improvement=False — suppresses message injection
- needs_improvement=None — no injection even when not satisfied
- Input data keys preserved in output
- send() passthrough
- get_executor_class() dispatcher
- _inject_improvement_message() direct unit tests
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from modules.reflection.node import ReflectionExecutor, get_executor_class


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_reflection_llm_response(
    satisfied: bool = True,
    reason: str = "Looks good",
    needs_improvement=None,
) -> dict:
    """Build a minimal LLM response that ReflectionExecutor can parse."""
    import json
    payload = {
        "satisfied": satisfied,
        "reason": reason,
        "needs_improvement": needs_improvement,
    }
    return {"choices": [{"message": {"role": "assistant", "content": json.dumps(payload)}}]}


def make_executor() -> ReflectionExecutor:
    """Create a ReflectionExecutor with a mocked LLMBridge."""
    executor = ReflectionExecutor()
    executor.llm = MagicMock()
    executor.llm.chat_completion = AsyncMock()
    return executor


def make_input(
    user_msg: str = "Explain Python",
    assistant_msg: str = "Python is a programming language.",
    extra: dict = None,
) -> dict:
    """Build a minimal input_data dict with messages and content."""
    data = {
        "messages": [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_msg},
        ],
        "content": assistant_msg,
    }
    if extra:
        data.update(extra)
    return data


# ---------------------------------------------------------------------------
# Basic satisfied / not-satisfied behaviour
# ---------------------------------------------------------------------------

class TestReflectionBasic:
    """Core satisfied / not-satisfied behaviour."""

    @pytest.mark.asyncio
    async def test_satisfied_response_sets_satisfied_true(self):
        """When LLM says satisfied=True, output satisfied should be True."""
        executor = make_executor()
        executor.llm.chat_completion = AsyncMock(
            return_value=make_reflection_llm_response(satisfied=True, reason="Perfect")
        )

        result = await executor.receive(make_input())

        assert result["satisfied"] is True
        assert result["reflection"]["satisfied"] is True
        assert result["reflection"]["reason"] == "Perfect"

    @pytest.mark.asyncio
    async def test_not_satisfied_sets_satisfied_false(self):
        """When LLM says satisfied=False, output satisfied should be False."""
        executor = make_executor()
        executor.llm.chat_completion = AsyncMock(
            return_value=make_reflection_llm_response(
                satisfied=False,
                reason="Too vague",
                needs_improvement="Add more detail",
            )
        )

        result = await executor.receive(make_input())

        assert result["satisfied"] is False
        assert result["reflection"]["satisfied"] is False
        assert result["reflection"]["needs_improvement"] == "Add more detail"

    @pytest.mark.asyncio
    async def test_no_messages_defaults_to_satisfied(self):
        """When there are no messages, reflection should default to satisfied=True."""
        executor = make_executor()

        result = await executor.receive({})

        assert result["satisfied"] is True
        assert result["reflection"]["satisfied"] is True
        assert result["reflection"]["reason"] == "No content to evaluate"
        executor.llm.chat_completion.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_content_no_user_message_defaults_to_satisfied(self):
        """When there is no user message, reflection should default to satisfied=True."""
        executor = make_executor()

        result = await executor.receive(
            {"messages": [{"role": "assistant", "content": "Hello"}]}
        )

        assert result["satisfied"] is True
        executor.llm.chat_completion.assert_not_called()

    @pytest.mark.asyncio
    async def test_none_input_defaults_to_satisfied(self):
        """None input should be treated as empty dict and default to satisfied=True."""
        executor = make_executor()

        result = await executor.receive(None)

        assert result["satisfied"] is True
        executor.llm.chat_completion.assert_not_called()

    @pytest.mark.asyncio
    async def test_reflection_key_always_present(self):
        """Output must always contain a 'reflection' dict."""
        executor = make_executor()
        executor.llm.chat_completion = AsyncMock(
            return_value=make_reflection_llm_response(satisfied=True)
        )

        result = await executor.receive(make_input())

        assert "reflection" in result
        assert isinstance(result["reflection"], dict)
        assert "satisfied" in result["reflection"]
        assert "reason" in result["reflection"]
        assert "needs_improvement" in result["reflection"]

    @pytest.mark.asyncio
    async def test_satisfied_key_present_at_top_level(self):
        """satisfied must be present at the top level for ConditionalRouter routing."""
        executor = make_executor()
        executor.llm.chat_completion = AsyncMock(
            return_value=make_reflection_llm_response(satisfied=True)
        )

        result = await executor.receive(make_input())

        assert "satisfied" in result


# ---------------------------------------------------------------------------
# Improvement message injection
# ---------------------------------------------------------------------------

class TestImprovementInjection:
    """Tests for improvement message injection into messages."""

    @pytest.mark.asyncio
    async def test_not_satisfied_injects_improvement_message(self):
        """When not satisfied and needs_improvement is set, a user message should be appended."""
        executor = make_executor()
        executor.llm.chat_completion = AsyncMock(
            return_value=make_reflection_llm_response(
                satisfied=False,
                needs_improvement="Be more concise",
            )
        )

        result = await executor.receive(make_input())

        messages = result["messages"]
        injected = [m for m in messages if m.get("role") == "user" and "needs improvement" in m.get("content", "").lower()]
        assert len(injected) == 1
        assert "Be more concise" in injected[0]["content"]

    @pytest.mark.asyncio
    async def test_satisfied_does_not_inject_message(self):
        """When satisfied, no improvement message should be appended."""
        executor = make_executor()
        executor.llm.chat_completion = AsyncMock(
            return_value=make_reflection_llm_response(satisfied=True)
        )

        input_data = make_input()
        original_message_count = len(input_data["messages"])

        result = await executor.receive(input_data)

        assert len(result["messages"]) == original_message_count

    @pytest.mark.asyncio
    async def test_inject_improvement_false_suppresses_injection(self):
        """inject_improvement=False should prevent message injection even when not satisfied."""
        executor = make_executor()
        executor.llm.chat_completion = AsyncMock(
            return_value=make_reflection_llm_response(
                satisfied=False,
                needs_improvement="Add examples",
            )
        )

        input_data = make_input()
        original_message_count = len(input_data["messages"])

        result = await executor.receive(input_data, config={"inject_improvement": False})

        assert len(result["messages"]) == original_message_count
        assert result["satisfied"] is False

    @pytest.mark.asyncio
    async def test_no_injection_when_needs_improvement_is_none(self):
        """When needs_improvement is None, no message should be injected even if not satisfied."""
        executor = make_executor()
        executor.llm.chat_completion = AsyncMock(
            return_value=make_reflection_llm_response(
                satisfied=False,
                needs_improvement=None,
            )
        )

        input_data = make_input()
        original_message_count = len(input_data["messages"])

        result = await executor.receive(input_data)

        assert len(result["messages"]) == original_message_count

    @pytest.mark.asyncio
    async def test_injected_message_appended_at_end(self):
        """The improvement message should be the last message in the list."""
        executor = make_executor()
        executor.llm.chat_completion = AsyncMock(
            return_value=make_reflection_llm_response(
                satisfied=False,
                needs_improvement="Provide code examples",
            )
        )

        result = await executor.receive(make_input())

        last_msg = result["messages"][-1]
        assert last_msg["role"] == "user"
        assert "Provide code examples" in last_msg["content"]

    @pytest.mark.asyncio
    async def test_original_messages_not_mutated(self):
        """The original input messages list must not be mutated."""
        executor = make_executor()
        executor.llm.chat_completion = AsyncMock(
            return_value=make_reflection_llm_response(
                satisfied=False,
                needs_improvement="More detail",
            )
        )

        input_data = make_input()
        original_messages = list(input_data["messages"])

        await executor.receive(input_data)

        assert input_data["messages"] == original_messages


# ---------------------------------------------------------------------------
# LLM error handling
# ---------------------------------------------------------------------------

class TestReflectionErrorHandling:
    """Tests for LLM failure and malformed response handling."""

    @pytest.mark.asyncio
    async def test_llm_exception_defaults_to_satisfied(self):
        """An exception from the LLM should default to satisfied=True (fail-safe)."""
        executor = make_executor()
        executor.llm.chat_completion = AsyncMock(
            side_effect=Exception("Connection refused")
        )

        result = await executor.receive(make_input())

        assert result["satisfied"] is True
        assert "Reflection error" in result["reflection"]["reason"]

    @pytest.mark.asyncio
    async def test_malformed_json_response_uses_default(self):
        """Malformed JSON from LLM should fall back to default reflection_result."""
        executor = make_executor()
        executor.llm.chat_completion = AsyncMock(
            return_value={"choices": [{"message": {"content": "not valid json {"}}]}
        )

        result = await executor.receive(make_input())

        # Falls back to default: satisfied=False, reason="Could not evaluate"
        assert result["reflection"]["reason"] == "Could not evaluate"
        assert result["satisfied"] is False

    @pytest.mark.asyncio
    async def test_none_llm_response_uses_default(self):
        """None response from LLM should fall back to default reflection_result."""
        executor = make_executor()
        executor.llm.chat_completion = AsyncMock(return_value=None)

        result = await executor.receive(make_input())

        assert result["reflection"]["reason"] == "Could not evaluate"

    @pytest.mark.asyncio
    async def test_empty_choices_falls_back_to_satisfied(self):
        """Empty choices list causes IndexError caught by exception handler → satisfied=True."""
        executor = make_executor()
        executor.llm.chat_completion = AsyncMock(
            return_value={"choices": []}
        )

        result = await executor.receive(make_input())

        # IndexError is caught by the outer except block → fail-safe satisfied=True
        assert result["satisfied"] is True
        assert "Reflection error" in result["reflection"]["reason"]

    @pytest.mark.asyncio
    async def test_js_comment_stripped_before_json_parse(self):
        """JavaScript-style comments in LLM response should be stripped before parsing."""
        executor = make_executor()
        executor.llm.chat_completion = AsyncMock(
            return_value={
                "choices": [{
                    "message": {
                        "content": (
                            '// reflection result\n'
                            '{"satisfied": true, "reason": "Good", "needs_improvement": null}'
                        )
                    }
                }]
            }
        )

        result = await executor.receive(make_input())

        assert result["satisfied"] is True
        assert result["reflection"]["reason"] == "Good"


# ---------------------------------------------------------------------------
# Input data preservation
# ---------------------------------------------------------------------------

class TestInputDataPreservation:
    """Tests that input_data keys are preserved in the output."""

    @pytest.mark.asyncio
    async def test_original_keys_preserved(self):
        """All original input_data keys should be present in the output."""
        executor = make_executor()
        executor.llm.chat_completion = AsyncMock(
            return_value=make_reflection_llm_response(satisfied=True)
        )

        input_data = make_input(extra={
            "session_id": "sess_99",
            "user_name": "Bob",
            "iterations": 3,
        })

        result = await executor.receive(input_data)

        assert result["session_id"] == "sess_99"
        assert result["user_name"] == "Bob"
        assert result["iterations"] == 3

    @pytest.mark.asyncio
    async def test_content_key_preserved(self):
        """The 'content' key from agent_loop output should be preserved."""
        executor = make_executor()
        executor.llm.chat_completion = AsyncMock(
            return_value=make_reflection_llm_response(satisfied=True)
        )

        result = await executor.receive(make_input(assistant_msg="My answer"))

        assert result["content"] == "My answer"

    @pytest.mark.asyncio
    async def test_messages_key_preserved_when_satisfied(self):
        """The 'messages' list should be preserved unchanged when satisfied."""
        executor = make_executor()
        executor.llm.chat_completion = AsyncMock(
            return_value=make_reflection_llm_response(satisfied=True)
        )

        input_data = make_input()
        result = await executor.receive(input_data)

        assert result["messages"] == input_data["messages"]


# ---------------------------------------------------------------------------
# _inject_improvement_message direct unit tests
# ---------------------------------------------------------------------------

class TestInjectImprovementMessageDirect:
    """Direct unit tests for the _inject_improvement_message helper."""

    def test_appends_user_message(self):
        """Should append a user message to the messages list."""
        executor = ReflectionExecutor.__new__(ReflectionExecutor)
        input_data = {
            "messages": [{"role": "user", "content": "Hi"}],
            "other_key": "value",
        }

        result = executor._inject_improvement_message(input_data, "Be more specific")

        assert len(result["messages"]) == 2
        assert result["messages"][-1]["role"] == "user"
        assert "Be more specific" in result["messages"][-1]["content"]

    def test_does_not_mutate_original(self):
        """Should not mutate the original input_data dict."""
        executor = ReflectionExecutor.__new__(ReflectionExecutor)
        original_messages = [{"role": "user", "content": "Hi"}]
        input_data = {"messages": original_messages}

        executor._inject_improvement_message(input_data, "Improve this")

        assert len(input_data["messages"]) == 1

    def test_preserves_other_keys(self):
        """Other keys in input_data should be preserved."""
        executor = ReflectionExecutor.__new__(ReflectionExecutor)
        input_data = {
            "messages": [{"role": "user", "content": "Hi"}],
            "session_id": "abc",
            "content": "Some content",
        }

        result = executor._inject_improvement_message(input_data, "Hint")

        assert result["session_id"] == "abc"
        assert result["content"] == "Some content"

    def test_message_contains_needs_improvement_text(self):
        """The injected message content should include the needs_improvement text."""
        executor = ReflectionExecutor.__new__(ReflectionExecutor)
        input_data = {"messages": []}

        result = executor._inject_improvement_message(input_data, "Add code examples")

        assert "Add code examples" in result["messages"][-1]["content"]
        assert "needs improvement" in result["messages"][-1]["content"].lower()


# ---------------------------------------------------------------------------
# send() passthrough
# ---------------------------------------------------------------------------

class TestReflectionSend:
    """Tests for the send() passthrough method."""

    @pytest.mark.asyncio
    async def test_send_returns_data_unchanged(self):
        """send() must return the data unchanged."""
        executor = make_executor()
        data = {"key": "value", "nested": {"a": 1}}
        result = await executor.send(data)
        assert result == data

    @pytest.mark.asyncio
    async def test_send_none_returns_none(self):
        """send() with None should return None."""
        executor = make_executor()
        result = await executor.send(None)
        assert result is None


# ---------------------------------------------------------------------------
# get_executor_class dispatcher
# ---------------------------------------------------------------------------

class TestReflectionGetExecutorClass:
    """Tests for the get_executor_class module-level dispatcher."""

    @pytest.mark.asyncio
    async def test_returns_reflection_executor_for_correct_id(self):
        """get_executor_class('reflection') should return ReflectionExecutor."""
        cls = await get_executor_class("reflection")
        assert cls is ReflectionExecutor

    @pytest.mark.asyncio
    async def test_returns_none_for_unknown_id(self):
        """get_executor_class with an unknown id should return None."""
        cls = await get_executor_class("unknown_node")
        assert cls is None

    @pytest.mark.asyncio
    async def test_executor_class_is_instantiable(self):
        """The returned class should be instantiable without arguments."""
        cls = await get_executor_class("reflection")
        instance = cls()
        assert isinstance(instance, ReflectionExecutor)


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
