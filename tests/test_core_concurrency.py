import pytest
import threading
import os
import json
import time
from unittest.mock import MagicMock
from core.flow_manager import FlowManager
from core.module_manager import ModuleManager

TEST_FLOWS_FILE = "test_flows_concurrency.json"

@pytest.fixture
def flow_manager():
    if os.path.exists(TEST_FLOWS_FILE):
        os.remove(TEST_FLOWS_FILE)
    manager = FlowManager(storage_file=TEST_FLOWS_FILE)
    yield manager
    if os.path.exists(TEST_FLOWS_FILE):
        os.remove(TEST_FLOWS_FILE)

def test_flow_manager_concurrency(flow_manager):
    """Test that flows can be saved concurrently without corruption."""
    
    def save_worker(i):
        flow_manager.save_flow(f"Flow {i}", [], [])

    threads = []
    for i in range(20):
        t = threading.Thread(target=save_worker, args=(i,))
        threads.append(t)
        t.start()
    
    for t in threads:
        t.join()
        
    # Verify all flows were saved
    flows = flow_manager.list_flows()
    # 20 new flows + 1 default flow = 21
    assert len(flows) == 21
    
    # Verify file integrity
    with open(TEST_FLOWS_FILE, "r") as f:
        data = json.load(f)
        assert len(data) == 21

@pytest.fixture
def mock_module_manager(tmp_path):
    # Setup a temp directory for modules
    modules_dir = tmp_path / "modules"
    modules_dir.mkdir()
    
    # Create a dummy module
    mod_dir = modules_dir / "test_mod"
    mod_dir.mkdir()
    with open(mod_dir / "module.json", "w") as f:
        json.dump({"id": "test_mod", "enabled": False, "config": {}}, f)
        
    app = MagicMock()
    manager = ModuleManager(app, modules_dir=str(modules_dir))
    return manager

def test_module_config_concurrency(mock_module_manager):
    """Test that module config updates are thread-safe."""
    
    def config_worker(i):
        mock_module_manager.update_module_config("test_mod", {"val": i})

    threads = []
    for i in range(20):
        t = threading.Thread(target=config_worker, args=(i,))
        threads.append(t)
        t.start()
    
    for t in threads:
        t.join()
        
    # Verify the config is valid JSON and has a value
    mod = mock_module_manager.modules["test_mod"]
    assert "val" in mod["config"]