import pytest
import threading
import os
import json
from unittest.mock import MagicMock, patch, AsyncMock
from core.settings import SettingsManager
from core.flow_runner import FlowRunner

TEST_SETTINGS_FILE = "test_settings_concurrency.json"

@pytest.fixture
def settings_manager():
    if os.path.exists(TEST_SETTINGS_FILE):
        os.remove(TEST_SETTINGS_FILE)
    manager = SettingsManager(file_path=TEST_SETTINGS_FILE)
    yield manager
    if os.path.exists(TEST_SETTINGS_FILE):
        os.remove(TEST_SETTINGS_FILE)

def test_settings_concurrency(settings_manager):
    """Test that settings can be updated concurrently without corruption."""
    
    def update_worker(key, value):
        for _ in range(10):
            settings_manager.save_settings({key: value})
    
    threads = []
    threads.append(threading.Thread(target=update_worker, args=("key1", "val1")))
    threads.append(threading.Thread(target=update_worker, args=("key2", "val2")))
    threads.append(threading.Thread(target=update_worker, args=("key3", "val3")))
    
    for t in threads:
        t.start()
    
    for t in threads:
        t.join()
        
    # Verify integrity
    assert settings_manager.get("key1") == "val1"
    assert settings_manager.get("key2") == "val2"
    assert settings_manager.get("key3") == "val3"
    
    # Verify file integrity
    with open(TEST_SETTINGS_FILE, "r") as f:
        data = json.load(f)
        assert data["key1"] == "val1"

@pytest.mark.asyncio
async def test_flow_runner_input_isolation():
    """Test that FlowRunner copies input for source nodes."""
    
    # Mock flow with two source nodes
    mock_flow = {
        "id": "test",
        "nodes": [
            {"id": "n1", "moduleId": "mod", "nodeTypeId": "type", "name": "Source 1"},
            {"id": "n2", "moduleId": "mod", "nodeTypeId": "type", "name": "Source 2"}
        ],
        "connections": []
    }
    
    with patch("core.flow_runner.flow_manager") as mock_fm, \
         patch("core.flow_runner.importlib.import_module") as mock_import:
        
        mock_fm.get_flow.return_value = mock_flow
        
        # Mock executor that modifies input in place
        class MutatingExecutor:
            async def receive(self, data, config=None):
                if isinstance(data, dict):
                    data["mutated"] = True
                return data
            async def send(self, data): return data
            
        mock_import.return_value.get_executor_class = AsyncMock(return_value=MutatingExecutor)
        
        runner = FlowRunner("test")
        initial_input = {"original": True}
        
        # Run
        await runner.run(initial_input)
        
        # Assert original input was NOT modified
        assert "mutated" not in initial_input
        assert initial_input["original"] is True