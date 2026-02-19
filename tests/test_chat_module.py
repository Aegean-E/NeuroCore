import pytest
from fastapi.testclient import TestClient
from main import app
from core.dependencies import get_llm_bridge
from core.llm import LLMBridge

@pytest.fixture
def client():
    """A TestClient that handles startup and shutdown events."""
    with TestClient(app) as c:
        yield c

def test_chat_gui_route(client):
    # Test if the chat module's GUI route is accessible
    response = client.get("/chat/gui")
    assert response.status_code == 200
    assert "AI Assistant" in response.text

def test_chat_send_route(client, httpx_mock):
    # Define a consistent, isolated URL for this test to avoid state pollution
    test_api_url = "http://test-chat-api.local/v1"

    # Override the dependency to ensure our test uses a predictable LLMBridge
    def get_test_llm_bridge():
        return LLMBridge(base_url=test_api_url)

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

    response = client.post("/chat/send", data={"message": "hello"})

    # Clean up the dependency override to not affect other tests
    app.dependency_overrides.clear()

    assert response.status_code == 200
    assert "hello" in response.text
    assert "Mocked AI Response" in response.text
