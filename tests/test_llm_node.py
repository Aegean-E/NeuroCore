import pytest
from unittest.mock import patch, AsyncMock
from modules.llm_module.node import LLMExecutor

@pytest.mark.asyncio
async def test_llm_executor_receive():
    """Test that the executor calls the bridge with correct messages."""
    executor = LLMExecutor()
    
    with patch("modules.llm_module.node.LLMBridge") as MockBridge:
        mock_bridge_instance = MockBridge.return_value
        mock_bridge_instance.chat_completion = AsyncMock(return_value={"choices": [{"message": {"content": "Hello"}}]})
        
        input_data = {"messages": [{"role": "user", "content": "Hi"}]}
        result = await executor.receive(input_data)
        
        assert result["choices"][0]["message"]["content"] == "Hello"
        mock_bridge_instance.chat_completion.assert_called_once()

@pytest.mark.asyncio
async def test_llm_executor_config_override():
    """Test that node configuration overrides input data."""
    executor = LLMExecutor()
    
    with patch("modules.llm_module.node.LLMBridge") as MockBridge:
        mock_bridge_instance = MockBridge.return_value
        mock_bridge_instance.chat_completion = AsyncMock(return_value={})
        
        input_data = {"messages": [{"role": "user", "content": "Hi"}]}
        config = {"model": "gpt-4", "temperature": 0.5}
        
        await executor.receive(input_data, config)
        
        call_kwargs = mock_bridge_instance.chat_completion.call_args.kwargs
        assert call_kwargs["model"] == "gpt-4"
        assert call_kwargs["temperature"] == 0.5