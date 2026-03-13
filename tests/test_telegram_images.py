import pytest
from unittest.mock import MagicMock, patch, AsyncMock, mock_open
from modules.telegram.service import TelegramService

@pytest.mark.asyncio
async def test_process_photo_message():
    """Test that photo messages are downloaded and converted to multimodal input."""
    service = TelegramService()
    service.bridge = MagicMock()
    service.bridge.send_message = AsyncMock()
    service.session_map = {"123": "sess-1"}
    
    # Mock dependencies
    with patch("modules.telegram.service.session_manager") as mock_sm, \
         patch("modules.telegram.service.FlowRunner") as MockRunner, \
         patch("modules.telegram.service.settings") as mock_settings, \
         patch("modules.telegram.service.os.remove") as mock_remove, \
         patch("builtins.open", mock_open(read_data=b"fake_image_data")):
        
        # Fix: Return a list so first element is "flow-1"
        mock_settings.get.return_value = ["flow-1"]
        
        # Mock session with history that includes the user message
        mock_session = {"history": [{"role": "user", "content": [{"type": "text", "text": "Look at this"}, {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,ZmFrZV9pbWFnZV9kYXRh"}}]}]}
        mock_sm.get_session.return_value = mock_session
        
        # Mock Bridge File Info & Download (get_file_info is awaited)
        service.bridge.get_file_info = AsyncMock(return_value={"file_path": "photos/image.jpg"})
        service.bridge.download_file = AsyncMock(return_value=True)
        
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
        
        # Verify Flow was called - use keyword argument
        MockRunner.assert_called_once_with(flow_id="flow-1")
        runner_instance.run.assert_called_once()
        
        # Verify session manager was used
        mock_sm.get_session.assert_called()

@pytest.mark.asyncio
async def test_process_photo_download_fail():
    """Test handling of failed image downloads."""
    service = TelegramService()
    service.bridge = MagicMock()
    service.bridge.get_file_info = AsyncMock(return_value={"file_path": "path"})
    service.bridge.download_file = AsyncMock(return_value=False)  # Fail download
    service.bridge.send_message = AsyncMock()

    await service.process_message({"chat_id": 123, "type": "photo", "photo": {"file_id": "f"}})

    service.bridge.send_message.assert_called_with("⚠️ Failed to download image.", 123)
