import pytest
from unittest.mock import MagicMock, patch, AsyncMock, mock_open
from modules.telegram.service import TelegramService

@pytest.mark.asyncio
async def test_process_photo_message():
    """Test that photo messages are downloaded and converted to multimodal input."""
    service = TelegramService()
    service.bridge = MagicMock()
    service.session_map = {"123": "sess-1"}
    
    # Mock dependencies
    with patch("modules.telegram.service.session_manager") as mock_sm, \
         patch("modules.telegram.service.FlowRunner") as MockRunner, \
         patch("modules.telegram.service.settings") as mock_settings, \
         patch("modules.telegram.service.os.remove") as mock_remove, \
         patch("builtins.open", mock_open(read_data=b"fake_image_data")):
        
        # Setup Mocks
        mock_settings.get.return_value = "flow-1"
        mock_sm.get_session.return_value = {"history": []}
        
        # Mock Bridge File Info & Download
        service.bridge.get_file_info.return_value = {"file_path": "photos/image.jpg"}
        service.bridge.download_file = MagicMock(return_value=True)
        
        # Mock Flow Runner
        runner_instance = MockRunner.return_value
        runner_instance.run = AsyncMock(return_value={"content": "I see an image"})
        
        # Input Message (Photo)
        msg = {
            "chat_id": 123,
            "type": "photo",
            "photo": {"file_id": "file-123"},
            "caption": "Look at this"
        }
        
        # Execute
        await service.process_message(msg)
        
        # Verify Download
        service.bridge.get_file_info.assert_called_with("file-123")
        service.bridge.download_file.assert_called()
        
        # Verify Session Update (Multimodal)
        mock_sm.add_message.assert_any_call("sess-1", "assistant", "I see an image")
        
        # Check user message structure
        args, _ = mock_sm.add_message.call_args_list[0]
        session_id, role, content = args
        assert session_id == "sess-1"
        assert role == "user"
        assert isinstance(content, list)
        assert content[0] == {"type": "text", "text": "Look at this"}
        assert content[1]["type"] == "image_url"
        assert "base64" in content[1]["image_url"]["url"]

@pytest.mark.asyncio
async def test_process_photo_download_fail():
    """Test handling of failed image downloads."""
    service = TelegramService()
    service.bridge = MagicMock()
    service.bridge.get_file_info.return_value = {"file_path": "path"}
    service.bridge.download_file = MagicMock(return_value=False) # Fail download
    
    await service.process_message({"chat_id": 123, "type": "photo", "photo": {"file_id": "f"}})
    
    service.bridge.send_message.assert_called_with("⚠️ Failed to download image.", 123)