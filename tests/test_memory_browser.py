import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
from main import app

@pytest.fixture
def client():
    with TestClient(app) as c:
        # Enable the module
        c.app.state.module_manager.enable_module("memory_browser")
        yield c

@pytest.fixture
def mock_store():
    with patch("modules.memory_browser.router.memory_store") as ms:
        yield ms

def test_browser_gui_route(client, mock_store):
    mock_store.browse.return_value = []
    response = client.get("/memory_browser/gui")
    assert response.status_code == 200
    assert "Search memories..." in response.text

def test_browser_list_route(client, mock_store):
    mock_store.browse.return_value = [
        {
            "id": 1, 
            "type": "FACT", 
            "subject": "User", 
            "text": "Test Memory", 
            "confidence": 1.0, 
            "created_at": 1234567890
        }
    ]
    
    response = client.get("/memory_browser/list?q=Test&type=FACT")
    assert response.status_code == 200
    assert "Test Memory" in response.text
    mock_store.browse.assert_called_with(search_text="Test", mem_type="FACT", limit=50)

def test_delete_memory_route(client, mock_store):
    response = client.delete("/memory_browser/delete/123")
    assert response.status_code == 200
    mock_store.delete_entry.assert_called_with(123)