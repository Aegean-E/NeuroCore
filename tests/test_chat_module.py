import pytest
from fastapi.testclient import TestClient
from main import app

@pytest.fixture
def client():
    """A TestClient that handles startup and shutdown events."""
    with TestClient(app) as c:
        yield c

def test_read_root(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "NeuroCore" in response.text

def test_chat_gui_route(client):
    # Test if the chat module's GUI route is accessible
    response = client.get("/chat/gui")
    assert response.status_code == 200
    assert "AI Assistant" in response.text

def test_chat_send_route(client, httpx_mock):
    # Mock the LLM call that the chat module makes
    mock_response = {
        "choices": [
            {"message": {"content": "Mocked AI Response"}}
        ]
    }
    httpx_mock.add_response(
        url="http://localhost:1234/v1/chat/completions",
        json=mock_response,
        method="POST"
    )
    
    response = client.post("/chat/send", data={"message": "hello"})
    assert response.status_code == 200
    assert "hello" in response.text
    assert "Mocked AI Response" in response.text
