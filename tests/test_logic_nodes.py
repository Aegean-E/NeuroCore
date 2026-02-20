import pytest
import asyncio
import time
from unittest.mock import patch, AsyncMock
from modules.logic.node import DelayExecutor, ScriptExecutor, RepeaterExecutor, ConditionalRouterExecutor

@pytest.mark.asyncio
async def test_delay_executor_valid():
    """Test that DelayExecutor waits for the specified time."""
    executor = DelayExecutor()
    start_time = time.time()
    
    # Use a small delay to keep tests fast but measurable
    delay = 0.1
    await executor.receive({"data": "test"}, config={"seconds": delay})
    
    elapsed = time.time() - start_time
    assert elapsed >= delay

@pytest.mark.asyncio
async def test_delay_executor_invalid_config():
    """Test that DelayExecutor handles invalid config gracefully."""
    executor = DelayExecutor()
    
    # Should default to 1.0s (or 0 if we passed negative and logic clamped it, 
    # but here we test the try/except block for non-numbers)
    # We won't wait 1s here to avoid slowing tests, just ensure no crash.
    # We'll patch asyncio.sleep to verify the call instead.
    
    # Just ensuring it doesn't raise unhandled exceptions. 
    # Actually, let's just run it with a negative number which should be clamped to 0
    await executor.receive({}, config={"seconds": -5})

@pytest.mark.asyncio
async def test_script_executor_modify_data():
    """Test that Python script can modify data."""
    executor = ScriptExecutor()
    input_data = {"count": 1}
    code = "result['count'] += 1\nresult['new_field'] = 'success'"
    
    result = await executor.receive(input_data, config={"code": code})
    
    assert result["count"] == 2
    assert result["new_field"] == "success"
    # Ensure original input wasn't mutated (it copies input)
    assert input_data["count"] == 1

@pytest.mark.asyncio
async def test_script_executor_error_handling():
    """Test that syntax errors in scripts are caught and returned as errors."""
    executor = ScriptExecutor()
    code = "print(undefined_variable)"
    
    result = await executor.receive({"data": "ok"}, config={"code": code})
    
    assert "error" in result
    assert "Script failed" in result["error"]
    assert "name 'undefined_variable' is not defined" in result["error"]

@pytest.mark.asyncio
async def test_repeater_executor_trigger():
    """Test that RepeaterExecutor schedules a new flow run."""
    executor = RepeaterExecutor()
    input_data = {"test": "data"}
    config = {"delay": 0.01, "max_repeats": 1, "_flow_id": "test-flow"}
    
    with patch("core.flow_runner.FlowRunner") as MockRunner:
        runner_instance = MockRunner.return_value
        runner_instance.run = AsyncMock()
        
        await executor.receive(input_data, config=config)
        # Give the background task a moment to run
        await asyncio.sleep(0.05)
        
        MockRunner.assert_called_with("test-flow")
        runner_instance.run.assert_called_once()
        # Check that repeat count was incremented in the next run's data
        args, _ = runner_instance.run.call_args
        assert args[0]["_repeat_count"] == 1

@pytest.mark.asyncio
async def test_conditional_router_tool_exists():
    """Test router detects when tool_results field exists."""
    executor = ConditionalRouterExecutor()
    
    input_data = {
        "messages": [],
        "tool_results": [{"name": "Weather", "result": "Sunny"}]
    }
    
    config = {
        "condition_type": "tool_exists",
        "check_field": "tool_results"
    }
    
    result = await executor.receive(input_data, config)
    assert result is not None
    assert result == input_data

@pytest.mark.asyncio
async def test_conditional_router_tool_not_exists():
    """Test router blocks flow when tool_results doesn't exist."""
    executor = ConditionalRouterExecutor()
    
    input_data = {
        "messages": [{"role": "user", "content": "Hello"}]
    }
    
    config = {
        "condition_type": "tool_exists",
        "check_field": "tool_results"
    }
    
    result = await executor.receive(input_data, config)
    assert result is None

@pytest.mark.asyncio
async def test_conditional_router_tool_results_count():
    """Test router checks for non-empty tool_results list."""
    executor = ConditionalRouterExecutor()
    
    config = {
        "condition_type": "tool_results",
        "check_field": "tool_results"
    }
    
    input_with_results = {
        "tool_results": [{"tool": "Weather", "output": "Sunny"}]
    }
    result = await executor.receive(input_with_results, config)
    assert result is not None
    
    input_without_results = {
        "tool_results": []
    }
    result = await executor.receive(input_without_results, config)
    assert result is None

@pytest.mark.asyncio
async def test_conditional_router_field_equals():
    """Test router compares field value."""
    executor = ConditionalRouterExecutor()
    
    config = {
        "condition_type": "field_equals",
        "check_field": "status",
        "expected_value": "ready"
    }
    
    input_matching = {"status": "ready", "data": "test"}
    result = await executor.receive(input_matching, config)
    assert result is not None
    
    input_not_matching = {"status": "busy", "data": "test"}
    result = await executor.receive(input_not_matching, config)
    assert result is None

@pytest.mark.asyncio
async def test_conditional_router_invert_condition():
    """Test router inverts condition when invert is True."""
    executor = ConditionalRouterExecutor()
    
    config = {
        "condition_type": "tool_exists",
        "check_field": "tool_results",
        "invert": True
    }
    
    input_with_tools = {
        "tool_results": [{"tool": "Weather"}]
    }
    result = await executor.receive(input_with_tools, config)
    assert result is None
    
    input_without_tools = {
        "messages": []
    }
    result = await executor.receive(input_without_tools, config)
    assert result is not None

@pytest.mark.asyncio
async def test_conditional_router_expression():
    """Test router evaluates custom Python expressions."""
    executor = ConditionalRouterExecutor()
    
    config = {
        "condition_type": "expression",
        "expression": "len(data.get('messages', [])) > 0"
    }
    
    input_with_messages = {
        "messages": [{"role": "user", "content": "Hi"}]
    }
    result = await executor.receive(input_with_messages, config)
    assert result is not None
    
    input_without_messages = {
        "messages": []
    }
    result = await executor.receive(input_without_messages, config)
    assert result is None

@pytest.mark.asyncio
async def test_conditional_router_none_input():
    """Test router returns None for None input."""
    executor = ConditionalRouterExecutor()
    result = await executor.receive(None, {})
    assert result is None