import pytest
from concurrent.futures import Future
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from main import app

@pytest.fixture
def client():
    with TestClient(app) as c:
        # Enable the module to ensure routes are mounted
        c.app.state.module_manager.enable_module("memory")
        yield c

@pytest.fixture
def mock_store():
    with patch("modules.memory.router.memory_store") as ms:
        mock_executor = MagicMock()
        ms.executor = mock_executor
        yield ms

def test_memory_stats_route(client, mock_store):
    """Test that the stats endpoint renders the correct HTML with data."""
    f = Future()
    f.set_result({
        "total": 10,
        "archived": 2,
        "user": 5,
        "assistant": 3,
        "grand_total": 12,
        "types": {
            "FACT": 8,
            "RULE": 2
        }
    })
    mock_store.executor.submit.return_value = f
    
    response = client.get("/memory/stats")
    assert response.status_code == 200
    assert "10" in response.text  # Total count
    assert "FACT" in response.text
    assert "RULE" in response.text
    assert "User Origin" in response.text

def test_memory_stats_empty(client, mock_store):
    """Test the empty state of the stats endpoint."""
    f = Future()
    f.set_result({"total": 0})
    mock_store.executor.submit.return_value = f
    
    response = client.get("/memory/stats")
    assert response.status_code == 200
    assert "No memories stored yet" in response.text