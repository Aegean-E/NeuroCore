"""
Tests for Agent Loop re-planning functionality
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from modules.agent_loop.node import AgentLoopExecutor


def make_executor():
    """Create an AgentLoopExecutor with patched LLMBridge."""
    with patch("modules.agent_loop.node.LLMBridge"):
        executor = AgentLoopExecutor()
    executor.llm = MagicMock()
    return executor


def make_input(messages=None, plan=None, current_step=0):
    """Create input data with optional plan."""
    data = {"messages": messages or [{"role": "user", "content": "Test"}]}
    if plan is not None:
        data["plan"] = plan
        data["current_step"] = current_step
    return data


@pytest.mark.asyncio
async def test_replan_needed_on_max_iterations():
    """Should set replan_needed=True when max_iterations reached."""
    executor = make_executor()
    
    # Simulate max iterations reached
    plan = [{"step": 1, "action": "Step 1"}, {"step": 2, "action": "Step 2"}]
    input_data = make_input(plan=plan, current_step=0)
    
    # Mock LLM to always return tool calls (forcing max iterations)
    # Tool should succeed so we don't get tool error instead of max iterations
    executor.llm.chat_completion = AsyncMock(return_value={
        "choices": [{
            "message": {
                "content": "Using tool",
                "tool_calls": [{"id": "1", "function": {"name": "TestTool", "arguments": "{}"}}]
            }
        }]
    })
    
    # Mock tool that succeeds
    with patch.object(executor, '_load_tool_library', return_value={
        "TestTool": "result = 'Success'"
    }):
        result = await executor.receive(input_data, config={"max_iterations": 2})
    
    assert result["replan_needed"] is True
    assert "Max iterations" in result["replan_reason"]
    assert "suggested_approach" in result



@pytest.mark.asyncio
async def test_replan_needed_on_tool_error():
    """Should set replan_needed=True when tool errors occur."""
    executor = make_executor()
    
    plan = [{"step": 1, "action": "Search"}]
    input_data = make_input(plan=plan, current_step=0)
    
    # Mock LLM to return a tool call
    executor.llm.chat_completion = AsyncMock(return_value={
        "choices": [{
            "message": {
                "content": "Using tool",
                "tool_calls": [{"id": "1", "function": {"name": "TestTool", "arguments": "{}"}}]
            }
        }]
    })
    
    # Tool library with failing tool
    with patch.object(executor, '_load_tool_library', return_value={
        "TestTool": "result = 'Error: Tool failed'"
    }):
        result = await executor.receive(input_data, config={"max_iterations": 5})
    
    assert result["replan_needed"] is True
    assert "Tool execution errors" in result["replan_reason"]


@pytest.mark.asyncio
async def test_no_replan_when_successful():
    """Should set replan_needed=False when execution succeeds."""
    executor = make_executor()
    
    plan = [{"step": 1, "action": "Simple task"}]
    input_data = make_input(plan=plan, current_step=0)
    
    # Mock LLM to return no tool calls (success)
    executor.llm.chat_completion = AsyncMock(return_value={
        "choices": [{"message": {"content": "Task completed successfully"}}]
    })
    
    with patch.object(executor, '_load_tool_library', return_value={}):
        result = await executor.receive(input_data, config={"max_iterations": 5})
    
    assert result["replan_needed"] is False
    assert result.get("replan_reason") is None


@pytest.mark.asyncio
async def test_replan_suggestion_for_multi_step_plan():
    """Should suggest breaking multi-step plans when re-planning."""
    executor = make_executor()
    
    plan = [
        {"step": 1, "action": "A"},
        {"step": 2, "action": "B"},
        {"step": 3, "action": "C"}
    ]
    input_data = make_input(plan=plan, current_step=1)
    
    # Force max iterations
    executor.llm.chat_completion = AsyncMock(return_value={
        "choices": [{
            "message": {
                "content": "Using tool",
                "tool_calls": [{"id": "1", "function": {"name": "Tool", "arguments": "{}"}}]
            }
        }]
    })
    
    with patch.object(executor, '_load_tool_library', return_value={}):
        result = await executor.receive(input_data, config={"max_iterations": 2})
    
    assert result["replan_needed"] is True
    assert "3 steps" in result["suggested_approach"]
    assert "breaking into smaller sub-tasks" in result["suggested_approach"]


@pytest.mark.asyncio
async def test_replan_suggestion_for_single_step_plan():
    """Should suggest different approach for single-step plan failure."""
    executor = make_executor()
    
    plan = [{"step": 1, "action": "Complex task"}]
    input_data = make_input(plan=plan, current_step=0)
    
    # Force max iterations
    executor.llm.chat_completion = AsyncMock(return_value={
        "choices": [{
            "message": {
                "content": "Using tool",
                "tool_calls": [{"id": "1", "function": {"name": "Tool", "arguments": "{}"}}]
            }
        }]
    })
    
    with patch.object(executor, '_load_tool_library', return_value={}):
        result = await executor.receive(input_data, config={"max_iterations": 2})
    
    assert result["replan_needed"] is True
    assert "Single-step plan failed" in result["suggested_approach"]


@pytest.mark.asyncio
async def test_replan_suggestion_when_no_plan():
    """Should suggest creating a plan when no plan exists."""
    executor = make_executor()
    
    input_data = make_input(plan=[], current_step=0)
    
    # Force max iterations
    executor.llm.chat_completion = AsyncMock(return_value={
        "choices": [{
            "message": {
                "content": "Using tool",
                "tool_calls": [{"id": "1", "function": {"name": "Tool", "arguments": "{}"}}]
            }
        }]
    })
    
    with patch.object(executor, '_load_tool_library', return_value={}):
        result = await executor.receive(input_data, config={"max_iterations": 2})
    
    assert result["replan_needed"] is True
    assert "No plan exists" in result["suggested_approach"]


@pytest.mark.asyncio
async def test_replan_on_timeout():
    """Should set replan_needed=True on timeout."""
    executor = make_executor()
    
    plan = [{"step": 1, "action": "Slow task"}]
    input_data = make_input(plan=plan, current_step=0)
    
    # Mock LLM to sleep longer than timeout
    async def slow_response(*args, **kwargs):
        await asyncio.sleep(10)
        return {"choices": [{"message": {"content": "Done"}}]}
    
    executor.llm.chat_completion = AsyncMock(side_effect=slow_response)
    
    with patch.object(executor, '_load_tool_library', return_value={}):
        result = await executor.receive(input_data, config={"timeout": 0.1})
    
    assert result["replan_needed"] is True
    assert "timed out" in result["replan_reason"].lower()


@pytest.mark.asyncio
async def test_replan_on_exception():
    """Should set replan_needed=True on unexpected exception."""
    executor = make_executor()
    
    plan = [{"step": 1, "action": "Task"}]
    input_data = make_input(plan=plan, current_step=0)
    
    # Mock _execute to raise an exception (simulating unexpected error in execution)
    with patch.object(executor, '_run_agent_loop', side_effect=Exception("Unexpected error")):
        with patch.object(executor, '_load_tool_library', return_value={}):
            result = await executor.receive(input_data, config={"max_iterations": 5})
    
    assert result["replan_needed"] is True
    assert "Unexpected error" in result["replan_reason"]



@pytest.mark.asyncio
async def test_replan_preserves_input_data():
    """Should preserve all input data fields when adding replan info."""
    executor = make_executor()
    
    input_data = {
        "messages": [{"role": "user", "content": "Test"}],
        "plan": [{"step": 1, "action": "Test"}],
        "current_step": 0,
        "session_id": "abc123",
        "custom_field": "value"
    }
    
    # Force max iterations with successful tool
    executor.llm.chat_completion = AsyncMock(return_value={
        "choices": [{
            "message": {
                "content": "Using tool",
                "tool_calls": [{"id": "1", "function": {"name": "Tool", "arguments": "{}"}}]
            }
        }]
    })
    
    with patch.object(executor, '_load_tool_library', return_value={
        "Tool": "result = 'Success'"
    }):
        result = await executor.receive(input_data, config={"max_iterations": 2})
    
    # Check that original fields are preserved
    assert result["session_id"] == "abc123"
    assert result["custom_field"] == "value"
    assert result["plan"] == input_data["plan"]
    # Messages will be modified by agent loop (appended to), which is expected behavior
    assert len(result["messages"]) >= len(input_data["messages"])



if __name__ == "__main__":
    import asyncio
    import pytest
    pytest.main([__file__, "-v"])
