import pytest
import sys
import importlib
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock, AsyncMock
from main import app


@pytest.fixture
def mock_chat_sessions():
    """Mocks the session manager to use an in-memory dict.
    
    This fixture must run AFTER the client fixture to ensure the router module
    is properly loaded. We need to patch the session_manager in sys.modules
    after the TestClient has been created (which triggers module loading in lifespan).
    """
    # First, ensure the router module is imported (this is done by client fixture)
    # We use sys.modules to get the actual module object (not the APIRouter)
    import modules.chat.router as router_module
    router_mod = sys.modules.get('modules.chat.router')
    
    # The router module may be either a module object or an APIRouter depending on 
    # import order. We need to get the actual module to patch session_manager.
    if router_mod is not None and hasattr(router_mod, 'session_manager'):
        target_session_manager = router_mod.session_manager
    else:
        # Fallback: try to get from sessions directly
        from modules.chat import sessions
        target_session_manager = sessions.session_manager
    
    mock_sessions_dict = {}
    target_session_manager.sessions = mock_sessions_dict
    target_session_manager._load_sessions = lambda: mock_sessions_dict
    target_session_manager._save_sessions = lambda: None
    mock_sessions_dict.clear()
    return target_session_manager


@pytest.fixture
def client(mock_chat_sessions):
    """Create test client with chat module enabled."""
    with TestClient(app) as c:
        c.app.state.module_manager.enable_module('chat')
        yield c


def test_send_message_no_active_flow(client, mock_chat_sessions):
    """Test response when no AI flow is active."""
    session = mock_chat_sessions.create_session("Test")
    
    def mock_settings_get(key, default=None):
        if key == "active_ai_flows":
            return []
        return default
    
    with patch("modules.chat.router.settings.get", side_effect=mock_settings_get):
        response = client.post(f"/chat/send?session_id={session['id']}", data={"message": "Hi"})
        
        assert response.status_code == 200
        assert "Error: No active AI Flow is set" in response.text

def test_send_message_flow_execution_error(client, mock_chat_sessions):
    """Test response when the AI flow returns an error."""
    session = mock_chat_sessions.create_session("Test")
    
    def mock_settings_get(key, default=None):
        if key == "active_ai_flows":
            return ["flow-1"]
        return default
    
    with patch("modules.chat.router.settings.get", side_effect=mock_settings_get), \
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
    
    def mock_settings_get(key, default=None):
        if key == "active_ai_flows":
            return ["flow-1"]
        return default
    
    with patch("modules.chat.router.settings.get", side_effect=mock_settings_get), \
         patch("modules.chat.router.flow_manager.get_flow", return_value={"id": "flow-1"}), \
         patch("modules.chat.router.FlowRunner") as MockRunner:
        
        runner_instance = MockRunner.return_value
        runner_instance.run = AsyncMock(side_effect=Exception("Critical Failure"))
        
        response = client.post(f"/chat/send?session_id={session['id']}", data={"message": "Hi"})
        
        assert response.status_code == 200
        assert "Critical Error running AI Flow" in response.text

def test_send_message_success(client, mock_chat_sessions):
    """Test successful message flow."""
    session = mock_chat_sessions.create_session("Test")
    
    def mock_settings_get(key, default=None):
        if key == "active_ai_flows":
            return ["flow-1"]
        return default
    
    with patch("modules.chat.router.settings.get", side_effect=mock_settings_get), \
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
