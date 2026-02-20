import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from modules.telegram.bridge import TelegramBridge
from modules.telegram.node import TelegramOutputExecutor
from modules.telegram.service import TelegramService

@pytest.fixture
def mock_requests():
    with patch("modules.telegram.bridge.requests.Session") as mock:
        session_instance = mock.return_value
        session_instance.post.return_value.status_code = 200
        session_instance.get.return_value.status_code = 200
        yield session_instance

def test_bridge_context_manager(mock_requests):
    """Test that the bridge closes the session when used as a context manager."""
    with TelegramBridge("token", 123) as bridge:
        pass
    mock_requests.close.assert_called_once()

def test_bridge_send_message_chunking(mock_requests):
    """Test that long messages are split into chunks."""
    bridge = TelegramBridge("token", 123)
    
    # Create a message longer than the 3072 limit
    long_message = "a" * 4000
    bridge.send_message(long_message)
    
    # Should be called twice (3072 + 928)
    assert mock_requests.post.call_count == 2

@pytest.mark.asyncio
async def test_output_executor_sends_message(mock_requests):
    """Test that the output node sends content via the bridge."""
    
    # Mock config loader
    with patch("modules.telegram.node.ConfigLoader.get_config", return_value={"bot_token": "t", "chat_id": 1}):
        executor = TelegramOutputExecutor()
        
        # Test with direct content
        await executor.receive({"content": "Hello"})
        mock_requests.post.assert_called()
        args, kwargs = mock_requests.post.call_args
        assert kwargs['json']['text'] == "Hello"

@pytest.mark.asyncio
async def test_service_process_message_flow_trigger():
    """Test that a text message triggers the AI flow."""
    service = TelegramService()
    service.bridge = MagicMock()
    
    # Mock dependencies
    with patch("modules.telegram.service.session_manager") as mock_sm, \
         patch("modules.telegram.service.FlowRunner") as MockRunner, \
         patch("modules.telegram.service.settings") as mock_settings:
        
        # Setup Mocks
        mock_settings.get.return_value = "flow-1"
        
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
        
        # Verify Flow Execution
        MockRunner.assert_called_with(flow_id="flow-1")
        runner_instance.run.assert_called_once()
        
        # Verify Reply
        service.bridge.send_message.assert_called_with("AI Reply", 123)
        assert mock_sm.add_message.called

@pytest.mark.asyncio
async def test_service_commands():
    """Test handling of telegram commands."""
    service = TelegramService()
    service.bridge = MagicMock()
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