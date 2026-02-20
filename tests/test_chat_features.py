import asyncio
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock, MagicMock
from main import app
from modules.chat import sessions
from core.dependencies import get_llm_bridge
from core.llm import LLMBridge

@pytest.fixture
def client():
    with TestClient(app) as c:
        c.app.state.module_manager.enable_module('chat')
        yield c

@pytest.fixture(autouse=True)
def mock_chat_sessions(monkeypatch):
    """Mocks the session manager to use an in-memory dict."""
    mock_sessions_dict = {}
    monkeypatch.setattr(sessions.session_manager, "sessions", mock_sessions_dict)
    monkeypatch.setattr(sessions.session_manager, "_load_sessions", lambda: mock_sessions_dict)
    monkeypatch.setattr(sessions.session_manager, "_save_sessions", lambda: None)
    mock_sessions_dict.clear()
    yield sessions.session_manager

@pytest.fixture(autouse=True)
def patch_settings_url(monkeypatch):
    """Ensures all components use localhost for LLM calls during tests."""
    from core.settings import settings as global_settings
    # Directly override the settings dictionary for reliability
    monkeypatch.setitem(global_settings.settings, "llm_api_url", "http://localhost:1234/v1")
    monkeypatch.setitem(global_settings.settings, "embedding_api_url", "http://localhost:1234/v1")
    monkeypatch.setitem(global_settings.settings, "default_model", "test-model")

@pytest.mark.asyncio
async def test_multimodal_image_upload(client, mock_chat_sessions, httpx_mock, monkeypatch):
    """Test sending an image file results in multimodal content structure."""
    
    # Mock LLM response (Flow only)
    httpx_mock.add_response(url="http://localhost:1234/v1/chat/completions", json={"choices": [{"message": {"content": "I see an image."}}]}, method="POST")
    
    # Mock the embedding call (Recall only)
    httpx_mock.add_response(url="http://localhost:1234/v1/embeddings", json={"data": [{"embedding": [0.1] * 1536}]}, method="POST")

    # Create session
    session = mock_chat_sessions.create_session("Image Test")
    
    # Mock file upload
    files = {'image': ('test.jpg', b'fake-image-bytes', 'image/jpeg')}
    data = {'message': 'Describe this', 'session_id': session['id']}
    
    # Override LLM dependency
    app.dependency_overrides[get_llm_bridge] = lambda: LLMBridge(base_url="http://localhost:1234/v1")
    
    # Patch create_task to avoid background tasks hanging or failing silently
    # Also patch _save_background to avoid extra LLM calls that consume mocks
    with patch("asyncio.create_task", side_effect=lambda coro: asyncio.ensure_future(coro)), \
         patch("modules.memory.node.MemorySaveExecutor._save_background", new_callable=AsyncMock):
        response = client.post(f"/chat/send?session_id={session['id']}", data={'message': 'Describe this'}, files=files)
        # Give background tasks a moment to run
        await asyncio.sleep(0.1)
    
    app.dependency_overrides.clear()
    
    assert response.status_code == 200
    
    # Verify session history structure
    history = session['history']
    user_msg = history[-2] # Second to last is user
    assert user_msg['role'] == 'user'
    assert isinstance(user_msg['content'], list)
    assert user_msg['content'][0] == {"type": "text", "text": "Describe this"}
    assert user_msg['content'][1]["type"] == "image_url"
    assert "data:image/jpeg;base64" in user_msg['content'][1]["image_url"]["url"]

@pytest.mark.asyncio
async def test_auto_rename_trigger(client, mock_chat_sessions, httpx_mock):
    """Test that session is renamed after N turns."""
    
    # Mock LLM responses (Flow then Rename)
    httpx_mock.add_response(url="http://localhost:1234/v1/chat/completions", json={"choices": [{"message": {"content": "Chat Response"}}]}, method="POST")
    httpx_mock.add_response(url="http://localhost:1234/v1/chat/completions", json={"choices": [{"message": {"content": "New Title"}}]}, method="POST")
    
    # Mock the embedding call (Recall only)
    httpx_mock.add_response(url="http://localhost:1234/v1/embeddings", json={"data": [{"embedding": [0.1] * 1536}]}, method="POST") # Recall
    
    # Setup session
    session = mock_chat_sessions.create_session("Session 2023-01-01")
    
    # Mock config to trigger rename after 1 turn (User+AI = 2 messages)
    with patch("modules.chat.router.session_manager.get_session") as mock_get_sess:
        # Instead, let's just rely on the default config or mock the module manager lookup
        # The default is 3 turns. Let's manually fill history to 5 messages so the next one makes it 6 (3 turns).
        session['history'] = [
            {"role": "user", "content": "1"}, {"role": "assistant", "content": "1"},
            {"role": "user", "content": "2"}, {"role": "assistant", "content": "2"},
            {"role": "user", "content": "3"} # Pending assistant response will make it 6
        ]
        
        mock_get_sess.return_value = session

        # We need to ensure the router logic sees this.
        # Since we can't easily mock the internal logic of the router function without complex patching,
        # we will verify the logic by checking if the LLM is called with the renaming prompt.
        
        # We'll use a spy on the LLMBridge.chat_completion method if possible, 
        # or just check if the session name changed if we can mock the second LLM call.
        
        # Let's add a second response for the renaming
        httpx_mock.add_response(
            url="http://localhost:1234/v1/chat/completions",
            json={"choices": [{"message": {"content": "New Title"}}]},
            method="POST"
        )

        app.dependency_overrides[get_llm_bridge] = lambda: LLMBridge(base_url="http://localhost:1234/v1")
        
        # Patch create_task to avoid background tasks hanging
        # Also patch _save_background to avoid extra LLM calls that consume mocks
        with patch("asyncio.create_task", side_effect=lambda coro: asyncio.ensure_future(coro)), \
             patch("modules.memory.node.MemorySaveExecutor._save_background", new_callable=AsyncMock):
            client.post(f"/chat/send?session_id={session['id']}", data={"message": "Trigger"})
            # Give background tasks a moment to run
            await asyncio.sleep(0.1)
        
        app.dependency_overrides.clear()
        
        # Check if name changed
        assert session['name'] == "New Title"