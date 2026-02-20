import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock, AsyncMock
from main import app
from modules.chat import sessions

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

def test_send_message_no_active_flow(client, mock_chat_sessions):
    """Test response when no AI flow is active."""
    session = mock_chat_sessions.create_session("Test")
    
    with patch("modules.chat.router.settings.get", return_value=None):
        response = client.post(f"/chat/send?session_id={session['id']}", data={"message": "Hi"})
        
        assert response.status_code == 200
        assert "Error: No active AI Flow is set" in response.text

def test_send_message_flow_execution_error(client, mock_chat_sessions):
    """Test response when the AI flow returns an error."""
    session = mock_chat_sessions.create_session("Test")
    
    with patch("modules.chat.router.settings.get", return_value="flow-1"), \
         patch("modules.chat.router.flow_manager.get_flow", return_value={"id": "flow-1"}), \
         patch("modules.chat.router.FlowRunner") as MockRunner:
        
        runner_instance = MockRunner.return_value
        runner_instance.run = AsyncMock(return_value={"error": "Something exploded"})
        
        response = client.post(f"/chat/send?session_id={session['id']}", data={"message": "Hi"})
        
        assert response.status_code == 200
        assert "Flow Execution Error: Something exploded" in response.text

def test_send_message_flow_crash(client, mock_chat_sessions):
    """Test response when the FlowRunner raises an exception."""
    session = mock_chat_sessions.create_session("Test")
    
    with patch("modules.chat.router.settings.get", return_value="flow-1"), \
         patch("modules.chat.router.flow_manager.get_flow", return_value={"id": "flow-1"}), \
         patch("modules.chat.router.FlowRunner") as MockRunner:
        
        runner_instance = MockRunner.return_value
        runner_instance.run = AsyncMock(side_effect=Exception("Critical Failure"))
        
        response = client.post(f"/chat/send?session_id={session['id']}", data={"message": "Hi"})
        
        assert response.status_code == 200
        assert "Critical Error running AI Flow: Critical Failure" in response.text

def test_send_message_success(client, mock_chat_sessions):
    """Test successful message flow."""
    session = mock_chat_sessions.create_session("Test")
    
    with patch("modules.chat.router.settings.get", return_value="flow-1"), \
         patch("modules.chat.router.flow_manager.get_flow", return_value={"id": "flow-1"}), \
         patch("modules.chat.router.FlowRunner") as MockRunner:
        
        runner_instance = MockRunner.return_value
        runner_instance.run = AsyncMock(return_value={"content": "AI Reply"})
        
        response = client.post(f"/chat/send?session_id={session['id']}", data={"message": "Hi"})
        
        assert response.status_code == 200
        assert "AI Reply" in response.text
        
        # Verify history update
        history = session["history"]
        assert len(history) == 2
        assert history[0]["content"] == "Hi"
        assert history[1]["content"] == "AI Reply"