import pytest
from modules.system_prompt.node import SystemPromptExecutor

@pytest.mark.asyncio
async def test_system_prompt_injection():
    executor = SystemPromptExecutor()
    
    input_data = {
        "messages": [
            {"role": "user", "content": "Hello"}
        ]
    }
    
    config = {"system_prompt": "You are a test bot."}
    
    result = await executor.receive(input_data, config)
    
    messages = result["messages"]
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "You are a test bot."
    assert messages[1]["role"] == "user"

@pytest.mark.asyncio
async def test_system_prompt_default():
    executor = SystemPromptExecutor()
    result = await executor.receive({"messages": []})
    assert "NeuroCore" in result["messages"][0]["content"]

@pytest.mark.asyncio
async def test_system_prompt_none_input():
    """Test handling of None input."""
    executor = SystemPromptExecutor()
    result = await executor.receive(None)
    
    assert "messages" in result
    assert len(result["messages"]) == 1
    assert result["messages"][0]["role"] == "system"

@pytest.mark.asyncio
async def test_system_prompt_preserves_data():
    """Test that other data fields are preserved."""
    executor = SystemPromptExecutor()
    input_data = {"other_field": 123, "messages": []}
    
    result = await executor.receive(input_data)
    assert result["other_field"] == 123
    assert len(result["messages"]) == 1

@pytest.mark.asyncio
async def test_system_prompt_bad_messages_type():
    """Test handling when messages is not a list."""
    executor = SystemPromptExecutor()
    # messages is None or a string, should be treated as empty list and prepended
    result = await executor.receive({"messages": None})
    assert len(result["messages"]) == 1
    assert result["messages"][0]["role"] == "system"