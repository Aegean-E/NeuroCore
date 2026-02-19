import os
import json
import pytest
from modules.chat.sessions import SessionManager

TEST_SESSIONS_FILE = "test_chat_sessions.json"

@pytest.fixture
def session_manager():
    """Provides a SessionManager instance using a temporary file."""
    # Setup
    if os.path.exists(TEST_SESSIONS_FILE):
        os.remove(TEST_SESSIONS_FILE)
    
    manager = SessionManager(storage_file=TEST_SESSIONS_FILE)
    yield manager
    
    # Teardown
    if os.path.exists(TEST_SESSIONS_FILE):
        os.remove(TEST_SESSIONS_FILE)

def test_create_and_get_session(session_manager):
    """Tests creating a session and retrieving it."""
    session_data = session_manager.create_session("My Test Session")
    session_id = session_data["id"]

    assert session_id in session_manager.sessions
    
    retrieved = session_manager.get_session(session_id)
    assert retrieved is not None
    assert retrieved["name"] == "My Test Session"
    assert retrieved["history"] == []

def test_add_message(session_manager):
    """Tests adding a message to a session's history."""
    session_data = session_manager.create_session()
    session_id = session_data["id"]

    result = session_manager.add_message(session_id, "user", "Hello")
    assert result is True
    
    session = session_manager.get_session(session_id)
    assert len(session["history"]) == 1
    assert session["history"][0] == {"role": "user", "content": "Hello"}

def test_delete_session(session_manager):
    """Tests deleting a session."""
    session_id = session_manager.create_session()["id"]
    assert session_manager.get_session(session_id) is not None
    
    result = session_manager.delete_session(session_id)
    assert result is True
    assert session_manager.get_session(session_id) is None

def test_list_sessions_sorted(session_manager):
    """Tests that sessions are listed with the newest first."""
    session1 = session_manager.create_session("Session 1")
    import time; time.sleep(0.01)
    session2 = session_manager.create_session("Session 2")
    
    session_list = session_manager.list_sessions()
    assert len(session_list) == 2
    assert session_list[0]["id"] == session2["id"]