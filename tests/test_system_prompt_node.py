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