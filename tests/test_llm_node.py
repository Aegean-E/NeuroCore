import pytest
from unittest.mock import patch, AsyncMock
from modules.llm_module.node import LLMExecutor, ConfigLoader

@pytest.mark.asyncio
async def test_llm_executor_receive():
    """Test that the executor calls the bridge with correct messages."""
    
    with patch("modules.llm_module.node.LLMBridge") as MockBridge:
        executor = LLMExecutor()
        mock_bridge_instance = MockBridge.return_value
        mock_bridge_instance.chat_completion = AsyncMock(return_value={"choices": [{"message": {"content": "Hello"}}]})
        
        input_data = {"messages": [{"role": "user", "content": "Hi"}]}
        result = await executor.receive(input_data)
        
        assert result["choices"][0]["message"]["content"] == "Hello"
        mock_bridge_instance.chat_completion.assert_called_once()

@pytest.mark.asyncio
async def test_llm_executor_config_override():
    """Test that node configuration overrides input data."""
    
    with patch("modules.llm_module.node.LLMBridge") as MockBridge:
        executor = LLMExecutor()
        mock_bridge_instance = MockBridge.return_value
        mock_bridge_instance.chat_completion = AsyncMock(return_value={})
        
        input_data = {"messages": [{"role": "user", "content": "Hi"}]}
        config = {"model": "gpt-4", "temperature": 0.5}
        
        await executor.receive(input_data, config)
        
        call_kwargs = mock_bridge_instance.chat_completion.call_args.kwargs
        assert call_kwargs["model"] == "gpt-4"
        assert call_kwargs["temperature"] == 0.5

@pytest.mark.asyncio
async def test_llm_executor_module_defaults():
    """Test that module defaults are used when no other config is present."""
    
    # Mock the ConfigLoader to return specific defaults
    with patch("modules.llm_module.node.ConfigLoader.get_config", return_value={"temperature": 0.9, "max_tokens": 100}):
        
        with patch("modules.llm_module.node.LLMBridge") as MockBridge:
            executor = LLMExecutor()
            mock_bridge_instance = MockBridge.return_value
            mock_bridge_instance.chat_completion = AsyncMock(return_value={})
            
            input_data = {"messages": [{"role": "user", "content": "Hi"}]}
            await executor.receive(input_data)
            
            call_kwargs = mock_bridge_instance.chat_completion.call_args.kwargs
            # Should use values from mocked module config
            assert call_kwargs["temperature"] == 0.9
            assert call_kwargs["max_tokens"] == 100

@pytest.mark.asyncio
async def test_llm_executor_input_override():
    """Test that input data overrides module defaults but not node config."""
    
    with patch("modules.llm_module.node.ConfigLoader.get_config", return_value={"temperature": 0.1}):
        
        with patch("modules.llm_module.node.LLMBridge") as MockBridge:
            executor = LLMExecutor()
            mock_bridge_instance = MockBridge.return_value
            mock_bridge_instance.chat_completion = AsyncMock(return_value={})
            
            # Input data has temperature 0.5
            input_data = {"messages": [{"role": "user", "content": "Hi"}], "temperature": 0.5}
            
            # Node config is empty
            await executor.receive(input_data, config={})
            
            call_kwargs = mock_bridge_instance.chat_completion.call_args.kwargs
            # Input data (0.5) should override module default (0.1)
            assert call_kwargs["temperature"] == 0.5

@pytest.mark.asyncio
async def test_llm_executor_receives_tools_from_system_prompt():
    """Test that LLM executor receives and passes tools from system prompt."""
    
    with patch("modules.llm_module.node.LLMBridge") as MockBridge:
        executor = LLMExecutor()
        mock_bridge_instance = MockBridge.return_value
        mock_bridge_instance.chat_completion = AsyncMock(return_value={"choices": [{"message": {"content": "Hello"}}]})
        
        # Input data includes tools from system prompt
        input_data = {
            "messages": [{"role": "user", "content": "Hi"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "Weather",
                        "description": "Get the current weather for a given location."
                    }
                }
            ]
        }
        
        result = await executor.receive(input_data)
        
        # Verify that tools were passed to the LLM
        call_kwargs = mock_bridge_instance.chat_completion.call_args.kwargs
        assert "tools" in call_kwargs
        assert len(call_kwargs["tools"]) == 1
        assert call_kwargs["tools"][0]["function"]["name"] == "Weather"
        assert call_kwargs["tool_choice"] == "auto"

@pytest.mark.asyncio
async def test_llm_executor_config_override_tools():
    """Test that node config can override tools from system prompt."""
    
    with patch("modules.llm_module.node.LLMBridge") as MockBridge:
        executor = LLMExecutor()
        mock_bridge_instance = MockBridge.return_value
        mock_bridge_instance.chat_completion = AsyncMock(return_value={})
        
        # Input data has tools from system prompt
        input_data = {
            "messages": [{"role": "user", "content": "Hi"}],
            "tools": [
                {
                    "type": "function",
                    "function": {"name": "Weather"}
                }
            ]
        }
        
        # Config overrides with different tools
        config = {
            "tools": [
                {
                    "type": "function",
                    "function": {"name": "Calculator"}
                }
            ]
        }
        
        await executor.receive(input_data, config)
        
        call_kwargs = mock_bridge_instance.chat_completion.call_args.kwargs
        # Config tools should override input_data tools
        assert call_kwargs["tools"][0]["function"]["name"] == "Calculator"