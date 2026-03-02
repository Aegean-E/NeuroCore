"""
Tests for modules/planner/node.py — PlanStepTracker
"""
import pytest
from modules.planner.node import PlanStepTracker, get_executor_class


@pytest.fixture
def executor():
    return PlanStepTracker()


@pytest.mark.asyncio
async def test_tracks_single_step_completion(executor):
    """Should increment current_step and mark step as completed."""
    data = {
        "plan": [{"step": 1, "action": "Search", "target": "docs"}],
        "current_step": 0
    }
    
    result = await executor.receive(data)
    
    assert result["current_step"] == 1
    assert result["step_completed"] == {"step": 1, "action": "Search", "target": "docs"}
    assert result["plan_complete"] is True


@pytest.mark.asyncio
async def test_tracks_multi_step_progress(executor):
    """Should track progress through multiple steps."""
    data = {
        "plan": [
            {"step": 1, "action": "Search", "target": "docs"},
            {"step": 2, "action": "Analyze", "target": "results"},
            {"step": 3, "action": "Report", "target": "findings"}
        ],
        "current_step": 0
    }
    
    # Step 1
    result = await executor.receive(data)
    assert result["current_step"] == 1
    assert result["step_completed"]["action"] == "Search"
    assert result["plan_complete"] is False
    
    # Step 2
    result = await executor.receive(result)
    assert result["current_step"] == 2
    assert result["step_completed"]["action"] == "Analyze"
    assert result["plan_complete"] is False
    
    # Step 3
    result = await executor.receive(result)
    assert result["current_step"] == 3
    assert result["step_completed"]["action"] == "Report"
    assert result["plan_complete"] is True


@pytest.mark.asyncio
async def test_resumes_from_middle_of_plan(executor):
    """Should work correctly when starting from non-zero current_step."""
    data = {
        "plan": [
            {"step": 1, "action": "A"},
            {"step": 2, "action": "B"},
            {"step": 3, "action": "C"}
        ],
        "current_step": 1  # Already completed step 1
    }
    
    result = await executor.receive(data)
    
    assert result["current_step"] == 2
    assert result["step_completed"]["action"] == "B"
    assert result["plan_complete"] is False


@pytest.mark.asyncio
async def test_empty_plan_passthrough(executor):
    """Should return data unchanged when plan is empty."""
    data = {"messages": [{"role": "user", "content": "Hi"}]}
    
    result = await executor.receive(data)
    
    assert result == data


@pytest.mark.asyncio
async def test_no_current_step_defaults_to_zero(executor):
    """Should default current_step to 0 if not provided."""
    data = {
        "plan": [{"step": 1, "action": "Test"}]
    }
    
    result = await executor.receive(data)
    
    assert result["current_step"] == 1
    assert result["step_completed"]["action"] == "Test"


@pytest.mark.asyncio
async def test_non_dict_input_passthrough(executor):
    """Should return non-dict input unchanged."""
    result = await executor.receive("not a dict")
    assert result == "not a dict"


@pytest.mark.asyncio
async def test_preserves_other_data_fields(executor):
    """Should preserve all other fields in input data."""
    data = {
        "plan": [{"step": 1, "action": "Test"}],
        "current_step": 0,
        "messages": [{"role": "user", "content": "Hi"}],
        "session_id": "abc123",
        "custom_field": "value"
    }
    
    result = await executor.receive(data)
    
    assert result["messages"] == [{"role": "user", "content": "Hi"}]
    assert result["session_id"] == "abc123"
    assert result["custom_field"] == "value"


@pytest.mark.asyncio
async def test_send_returns_data_unchanged(executor):
    """send() should return data unchanged."""
    data = {"test": "value"}
    result = await executor.send(data)
    assert result is data


@pytest.mark.asyncio
async def test_get_executor_class_returns_tracker():
    """get_executor_class('plan_step_tracker') should return PlanStepTracker."""
    cls = await get_executor_class("plan_step_tracker")
    assert cls is PlanStepTracker


@pytest.mark.asyncio
async def test_plan_complete_only_when_all_steps_done(executor):
    """plan_complete should only be True when all steps are finished."""
    plan = [
        {"step": 1, "action": "A"},
        {"step": 2, "action": "B"}
    ]
    
    # After first step
    result = await executor.receive({"plan": plan, "current_step": 0})
    assert result["plan_complete"] is False
    
    # After second step
    result = await executor.receive(result)
    assert result["plan_complete"] is True
    
    # After trying to go beyond
    result = await executor.receive(result)
    # Should stay complete and not crash
    assert result["plan_complete"] is True


@pytest.mark.asyncio
async def test_step_completed_field_set_correctly(executor):
    """step_completed should always reflect the step that was just completed."""
    plan = [
        {"step": 1, "action": "Search", "target": "docs"},
        {"step": 2, "action": "Analyze", "target": "data"}
    ]
    
    result = await executor.receive({"plan": plan, "current_step": 0})
    assert result["step_completed"] == plan[0]
    
    result = await executor.receive(result)
    assert result["step_completed"] == plan[1]


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
