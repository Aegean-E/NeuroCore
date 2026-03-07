import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from modules.telegram.bridge import TelegramBridge
from modules.telegram.node import TelegramOutputExecutor
from modules.telegram.service import TelegramService

@pytest.fixture
def mock_httpx():
    """Mock httpx.AsyncClient for async bridge tests."""
    with patch("modules.telegram.bridge.httpx.AsyncClient") as mock:
        client_instance = AsyncMock()
        mock.return_value = client_instance
        # Setup response methods
        client_instance.post = AsyncMock()
        client_instance.get = AsyncMock()
        client_instance.stream = AsyncMock()
        client_instance.post.return_value = AsyncMock(status_code=200)
        client_instance.get.return_value = AsyncMock(status_code=200)
        yield client_instance

@pytest.mark.asyncio
async def test_bridge_context_manager():
    """Test that the bridge closes the client when used as a context manager."""
    bridge = TelegramBridge("token", 123)
    async with bridge:
        pass
    # Should have closed the client
    assert bridge._client is None or bridge._client.is_closed

@pytest.mark.asyncio
async def test_bridge_send_message_chunking(mock_httpx):
    """Test that long messages are split into chunks."""
    bridge = TelegramBridge("token", 123)
    bridge._client = mock_httpx
    
    # Create a message longer than the 3072 limit
    long_message = "a" * 4000
    await bridge.send_message(long_message)
    
    # Should be called twice (3072 + 928)
    assert mock_httpx.post.call_count == 2

@pytest.mark.asyncio
async def test_output_executor_sends_message(mock_httpx):
    """Test that the output node sends content via the bridge."""
    
    # Mock config loader
    with patch("modules.telegram.node.ConfigLoader.get_config", return_value={"bot_token": "t", "chat_id": 1}):
        executor = TelegramOutputExecutor()
        
        # Test with direct content - need to mock the bridge
        executor._bridge_cache = {}
        mock_bridge = AsyncMock()
        executor._bridge_cache["t_1"] = mock_bridge
        
        await executor.receive({"content": "Hello"})
        mock_bridge.send_message.assert_called_once_with("Hello")

@pytest.mark.asyncio
async def test_service_process_message_flow_trigger():
    """Test that a text message triggers the AI flow."""
    service = TelegramService()
    service.bridge = AsyncMock()
    
    # Mock dependencies
    with patch("modules.telegram.service.session_manager") as mock_sm, \
         patch("modules.telegram.service.FlowRunner") as MockRunner, \
         patch("modules.telegram.service.settings") as mock_settings:
        
        # Setup Mocks - fix: return a list so first element is extracted
        mock_settings.get.return_value = ["flow-1"]
        
        # Session Manager
        mock_sm.get_session.return_value = {"history": []}
        mock_sm.create_session.return_value = {"id": "sess-1"}
        
        # Flow Runner
        runner_instance = MockRunner.return_value
        runner_instance.run = AsyncMock(return_value={"content": "AI Reply"})
        
        # Input Message
        msg = {
            "chat_id": 123,
            "type": "text",
            "text": "Hello AI"
        }
        
        # Execute
        await service.process_message(msg)
        
        # Verify Flow Execution - should extract first from list
        MockRunner.assert_called_with(flow_id="flow-1")
        runner_instance.run.assert_called_once()
        
        # Verify Reply - check that it was called with a message containing "AI Reply" and chat_id 123
        calls = service.bridge.send_message.call_args_list
        assert any(call[0][0] == "AI Reply" or call[0][0].startswith("AI Reply") for call in calls), f"Expected 'AI Reply' call, got {calls}"
        assert any(call[0][1] == 123 for call in calls), f"Expected chat_id 123, got {calls}"
        assert mock_sm.add_message.called

@pytest.mark.asyncio
async def test_service_commands():
    """Test handling of telegram commands."""
    service = TelegramService()
    service.bridge = AsyncMock()
    service.session_map = {}
    
    with patch("modules.telegram.service.session_manager") as mock_sm:
        mock_sm.create_session.return_value = {"id": "new-sess"}
        
        # Test /new_session
        msg = {"chat_id": 456, "type": "text", "text": "/new_session"}
        await service.process_message(msg)
        
        assert service.session_map["456"] == "new-sess"
        service.bridge.send_message.assert_called_with("🆕 New session started.", 456)
        
        # Test /help
        msg = {"chat_id": 456, "type": "text", "text": "/help"}
        await service.process_message(msg)
        # Check that help text was sent (checking substring)
        args, _ = service.bridge.send_message.call_args
        assert "NeuroCore Bridge Commands" in args[0]

