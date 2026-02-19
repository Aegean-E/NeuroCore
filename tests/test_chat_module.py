import pytest
from fastapi.testclient import TestClient
from main import app
from core.dependencies import get_llm_bridge
from core.llm import LLMBridge
from modules.chat import sessions

@pytest.fixture
def client():
    """A TestClient that handles startup and shutdown events."""
    # The chat module is disabled by default, so we must enable it for these tests.
    with TestClient(app) as c:
        # Access the running app's state to enable the module programmatically
        module_manager = c.app.state.module_manager
        module_manager.enable_module('chat')
        yield c

@pytest.fixture(autouse=True)
def mock_chat_sessions(monkeypatch):
    """Mocks the session manager to use an in-memory dict instead of a file."""
    mock_sessions_dict = {}
    
    def mock_load():
        return mock_sessions_dict

    def mock_save():
        pass # Do nothing, keep it in memory

    # Replace the manager's internal state and file operations
    monkeypatch.setattr(sessions.session_manager, "sessions", mock_sessions_dict)
    monkeypatch.setattr(sessions.session_manager, "_load_sessions", mock_load)
    monkeypatch.setattr(sessions.session_manager, "_save_sessions", mock_save)
    
    # Ensure the dictionary is clear before each test
    mock_sessions_dict.clear()
    yield sessions.session_manager


def test_chat_gui_route(client):
    # Test if the chat module's GUI route is accessible
    response = client.get("/chat/gui")
    assert response.status_code == 200
    assert "No chat session selected" in response.text

def test_get_chat_sessions_route(client, mock_chat_sessions):
    """Tests the endpoint that lists all chat sessions."""
    mock_chat_sessions.create_session("Test Session 1")
    response = client.get("/chat/sessions")
    assert response.status_code == 200
    assert "Test Session 1" in response.text

def test_chat_send_route(client, httpx_mock, mock_chat_sessions, monkeypatch):
    # Define a consistent, isolated URL for this test to avoid state pollution
    test_api_url = "http://test-chat-api.local/v1"

    # Override the dependency to ensure our test uses a predictable LLMBridge
    def get_test_llm_bridge():
        return LLMBridge(base_url=test_api_url)

    # Patch the global settings that the FlowRunner will use to create its own LLMBridge
    from core.settings import settings as global_settings
    monkeypatch.setitem(global_settings.settings, "llm_api_url", test_api_url)

    app.dependency_overrides[get_llm_bridge] = get_test_llm_bridge

    # Mock the LLM call that the chat module makes
    mock_response = {
        "choices": [
            {"message": {"content": "Mocked AI Response"}}
        ]
    }
    httpx_mock.add_response(
        url=f"{test_api_url}/chat/completions",
        json=mock_response,
        method="POST"
    )
    
    # Mock the embedding call that MemorySaveExecutor might trigger
    httpx_mock.add_response(
        url=f"{test_api_url}/embeddings",
        json={"data": [{"embedding": [0.1] * 1536}]},
        method="POST"
    )

    # Create a session to send a message to
    session = mock_chat_sessions.create_session("Send Test")
    session_id = session['id']

    response = client.post(f"/chat/send?session_id={session_id}", data={"message": "hello"})

    # Clean up the dependency override to not affect other tests
    app.dependency_overrides.clear()

    assert response.status_code == 200
    assert "hello" in response.text
    assert "Mocked AI Response" in response.text

def test_create_new_session_route(client, mock_chat_sessions):
    """Tests the endpoint for creating a new session."""
    assert len(mock_chat_sessions.list_sessions()) == 0
    response = client.post("/chat/sessions/new")
    assert response.status_code == 200
    assert "HX-Trigger" in response.headers
    assert "newSessionCreated" in response.headers["HX-Trigger"]
    assert len(mock_chat_sessions.list_sessions()) == 1

def test_delete_session_route(client, mock_chat_sessions):
    """Tests the endpoint for deleting a session."""
    session = mock_chat_sessions.create_session("To Be Deleted")
    session_id = session['id']
    assert len(mock_chat_sessions.list_sessions()) == 1

    response = client.post(f"/chat/sessions/{session_id}/delete", headers={"HX-Trigger": "sessionsChanged"})
    assert response.status_code == 200
    assert "sessionsChanged" in response.headers["HX-Trigger"]
    assert len(mock_chat_sessions.list_sessions()) == 0, "Deleting the last session should result in zero sessions"

def test_rename_session_route(client, mock_chat_sessions):
    """Tests the endpoint for renaming a session."""
    session = mock_chat_sessions.create_session("Old Name")
    session_id = session['id']
    
    response = client.post(f"/chat/sessions/{session_id}/rename", data={"name": "New Name"})
    assert response.status_code == 200
    assert "sessionsChanged" in response.headers["HX-Trigger"]
    
    renamed_session = mock_chat_sessions.get_session(session_id)
    assert renamed_session['name'] == "New Name"

def test_chat_page_route(client):
    """Tests if the main chat page loads correctly."""
    response = client.get("/chat")
    assert response.status_code == 200
    assert "NeuroCore" in response.text
    assert "Chat Sessions" in response.text
