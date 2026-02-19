import pytest
from modules.chat.node import ChatInputExecutor, ChatOutputExecutor

@pytest.mark.asyncio
async def test_chat_input_send_validation():
    """Test that Chat Input requires 'messages' in the data."""
    executor = ChatInputExecutor()
    
    # Valid case
    res = await executor.send({"messages": []})
    assert "messages" in res
    
    # Invalid case
    res = await executor.send({"other": "data"})
    assert "error" in res

@pytest.mark.asyncio
async def test_chat_output_receive_parsing():
    """Test that Chat Output correctly parses OpenAI-style responses."""
    executor = ChatOutputExecutor()
    
    # Standard OpenAI format
    input_data = {
        "choices": [
            {"message": {"content": "Response"}}
        ]
    }
    res = await executor.receive(input_data)
    assert res["content"] == "Response"
    
    # Error propagation (if previous node failed)
    res = await executor.receive({"error": "Previous error"})
    assert "error" in res
    assert res["error"] == "Previous error"
    
    # Malformed data (empty choices list triggers IndexError)
    res = await executor.receive({"choices": []})
    assert "error" in res
    assert "Could not parse" in res["error"]