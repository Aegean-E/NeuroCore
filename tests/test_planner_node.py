"""
Tests for modules/planner/node.py — PlannerExecutor
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from modules.planner.node import PlannerExecutor, get_executor_class


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_executor() -> PlannerExecutor:
    """Create a PlannerExecutor with LLMBridge patched out."""
    with patch("modules.planner.node.LLMBridge"):
        executor = PlannerExecutor()
    executor.llm = MagicMock()
    return executor


def make_llm_response(content: str) -> dict:
    return {"choices": [{"message": {"content": content}}]}


def make_input(user_msg: str) -> dict:
    return {"messages": [{"role": "user", "content": user_msg}]}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_none_input_returns_empty_dict():
    """None input should be treated as empty dict and returned with plan keys."""
    executor = make_executor()
    executor.llm.chat_completion = AsyncMock(return_value=make_llm_response("[]"))
    result = await executor.receive(None)
    # No user message → passthrough
    assert result == {}


@pytest.mark.asyncio
async def test_no_user_message_passthrough():
    """Input with no user message should be returned unchanged."""
    executor = make_executor()
    input_data = {"messages": [{"role": "assistant", "content": "Hello"}]}
    result = await executor.receive(input_data)
    assert result is input_data


@pytest.mark.asyncio
async def test_enabled_false_passthrough():
    """When enabled=False in config, input should be returned unchanged."""
    executor = make_executor()
    input_data = make_input("Do something")
    result = await executor.receive(input_data, config={"enabled": False})
    assert result is input_data


@pytest.mark.asyncio
async def test_valid_json_list_plan_parsed():
    """A valid JSON list response should be parsed into plan steps."""
    executor = make_executor()
    plan_json = '[{"step": 1, "action": "Search", "target": "Python docs"}]'
    executor.llm.chat_completion = AsyncMock(return_value=make_llm_response(plan_json))

    result = await executor.receive(make_input("Research Python"))

    assert result["plan_needed"] is True
    assert len(result["plan"]) == 1
    assert result["plan"][0]["action"] == "Search"
    assert result["plan"][0]["target"] == "Python docs"
    assert result["current_step"] == 0
    assert result["original_request"] == "Research Python"


@pytest.mark.asyncio
async def test_dict_wrapped_plan_parsed():
    """A plan wrapped in a dict ({"plan": [...]}) should also be parsed."""
    executor = make_executor()
    plan_json = '{"plan": [{"step": 1, "action": "Fetch", "target": "URL"}]}'
    executor.llm.chat_completion = AsyncMock(return_value=make_llm_response(plan_json))

    result = await executor.receive(make_input("Fetch a URL"))

    assert result["plan_needed"] is True
    assert result["plan"][0]["action"] == "Fetch"


@pytest.mark.asyncio
async def test_plan_context_formatted():
    """plan_context should be a formatted string of steps."""
    executor = make_executor()
    plan_json = '[{"step": 1, "action": "Step one", "target": "target A"}, {"step": 2, "action": "Step two", "target": "target B"}]'
    executor.llm.chat_completion = AsyncMock(return_value=make_llm_response(plan_json))

    result = await executor.receive(make_input("Do two things"))

    assert "plan_context" in result
    assert "## Execution Plan" in result["plan_context"]
    assert "Step one" in result["plan_context"]
    assert "Step two" in result["plan_context"]


@pytest.mark.asyncio
async def test_llm_failure_returns_empty_plan_with_error():
    """When LLM raises an exception, plan should be empty and planning_error set."""
    executor = make_executor()
    executor.llm.chat_completion = AsyncMock(side_effect=Exception("LLM unavailable"))

    result = await executor.receive(make_input("Do something"))

    assert result["plan"] == []
    assert result["plan_needed"] is False
    assert "planning_error" in result
    assert "LLM unavailable" in result["planning_error"]


@pytest.mark.asyncio
async def test_malformed_json_returns_empty_plan():
    """Malformed JSON in LLM response should result in empty plan (no crash)."""
    executor = make_executor()
    executor.llm.chat_completion = AsyncMock(return_value=make_llm_response("not valid json"))

    result = await executor.receive(make_input("Do something"))

    assert result["plan"] == []
    assert result["plan_needed"] is False


@pytest.mark.asyncio
async def test_max_steps_respected():
    """Plan should be truncated to max_steps."""
    executor = make_executor()
    steps = [{"step": i, "action": f"Action {i}", "target": ""} for i in range(1, 11)]
    executor.llm.chat_completion = AsyncMock(return_value=make_llm_response(
        str(steps).replace("'", '"')
    ))

    result = await executor.receive(make_input("Do 10 things"), config={"max_steps": 3})

    assert len(result["plan"]) <= 3


@pytest.mark.asyncio
async def test_input_data_preserved_in_output():
    """Original input_data keys should be preserved in the output."""
    executor = make_executor()
    executor.llm.chat_completion = AsyncMock(return_value=make_llm_response("[]"))

    input_data = {
        "messages": [{"role": "user", "content": "Hi"}],
        "session_id": "sess_1",
        "custom": "value",
    }
    result = await executor.receive(input_data)

    assert result["session_id"] == "sess_1"
    assert result["custom"] == "value"


@pytest.mark.asyncio
async def test_get_executor_class_dispatcher():
    """get_executor_class('planner') should return PlannerExecutor."""
    cls = await get_executor_class("planner")
    assert cls is PlannerExecutor


@pytest.mark.asyncio
async def test_get_executor_class_unknown():
    """get_executor_class with unknown id should return None."""
    cls = await get_executor_class("unknown")
    assert cls is None


# ============ NEW TESTS FOR MAX_TOKENS CALCULATION ============

@pytest.mark.asyncio
async def test_max_tokens_calculated_dynamically():
    """max_tokens should be calculated based on max_steps."""
    executor = make_executor()
    
    # Capture the actual call to chat_completion
    call_args = {}
    async def capture_call(*args, **kwargs):
        call_args.update(kwargs)
        return make_llm_response("[]")
    
    executor.llm.chat_completion = AsyncMock(side_effect=capture_call)

    # Test with max_steps=5
    await executor.receive(make_input("Test"), config={"max_steps": 5})
    
    # Expected: base_tokens(200) + tokens_per_step(50) * max_steps(5) = 450
    # But min_tokens is 500, so should be 500
    assert call_args["max_tokens"] == 500

    # Test with max_steps=20
    call_args.clear()
    await executor.receive(make_input("Test"), config={"max_steps": 20})
    
    # Expected: 200 + 50*20 = 1200
    assert call_args["max_tokens"] == 1200


@pytest.mark.asyncio
async def test_max_tokens_minimum_enforced():
    """max_tokens should never go below minimum (500)."""
    executor = make_executor()
    
    call_args = {}
    async def capture_call(*args, **kwargs):
        call_args.update(kwargs)
        return make_llm_response("[]")
    
    executor.llm.chat_completion = AsyncMock(side_effect=capture_call)

    # Test with max_steps=1 (very small plan)
    await executor.receive(make_input("Test"), config={"max_steps": 1})
    
    # Expected: 200 + 50*1 = 250, but min is 500
    assert call_args["max_tokens"] == 500


# ============ NEW TESTS FOR PLAN_CONTEXT CURRENT STEP ============

@pytest.mark.asyncio
async def test_plan_context_shows_current_step():
    """plan_context should indicate which step is current."""
    executor = make_executor()
    plan_json = '[{"step": 1, "action": "First", "target": "A"}, {"step": 2, "action": "Second", "target": "B"}]'
    executor.llm.chat_completion = AsyncMock(return_value=make_llm_response(plan_json))

    result = await executor.receive(make_input("Do two things"))

    assert "Currently on step 1 of 2" in result["plan_context"]
    assert "→ 1. First: A (CURRENT)" in result["plan_context"]
    assert "  2. Second: B" in result["plan_context"]


@pytest.mark.asyncio
async def test_plan_context_with_depends_on():
    """plan_context should preserve depends_on field in plan."""
    executor = make_executor()
    plan_json = '[{"step": 1, "action": "Prepare"}, {"step": 2, "action": "Execute", "depends_on": 1}]'
    executor.llm.chat_completion = AsyncMock(return_value=make_llm_response(plan_json))

    result = await executor.receive(make_input("Do something"))

    assert result["plan_needed"] is True
    assert len(result["plan"]) == 2
    assert result["plan"][1]["depends_on"] == 1


@pytest.mark.asyncio
async def test_plan_context_empty_plan():
    """plan_context should not be set for empty plans."""
    executor = make_executor()
    executor.llm.chat_completion = AsyncMock(return_value=make_llm_response("[]"))

    result = await executor.receive(make_input("Simple question"))

    assert result["plan_needed"] is False
    assert "plan_context" not in result or result["plan_context"] is None


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
