import os
import json
import pytest
from core.flow_manager import FlowManager

TEST_FLOWS_FILE = "test_flows.json"

@pytest.fixture
def flow_manager():
    """Provides a FlowManager instance using a temporary file, ensuring test isolation."""
    # Setup
    if os.path.exists(TEST_FLOWS_FILE):
        os.remove(TEST_FLOWS_FILE)
    
    manager = FlowManager(storage_file=TEST_FLOWS_FILE)
    yield manager
    
    # Teardown
    if os.path.exists(TEST_FLOWS_FILE):
        os.remove(TEST_FLOWS_FILE)

def test_initialization_creates_file(flow_manager):
    """Tests that a new, empty flow file is created on first run."""
    assert os.path.exists(TEST_FLOWS_FILE)
    assert flow_manager.flows == {}

def test_save_and_get_flow(flow_manager):
    """Tests saving a flow and retrieving it."""
    flow_data = flow_manager.save_flow("My Test Flow", [{"id": "n1"}], [{"from": "n1", "to": "n2"}])
    flow_id = flow_data["id"]

    assert flow_id in flow_manager.flows
    
    retrieved_flow = flow_manager.get_flow(flow_id)
    assert retrieved_flow is not None
    assert retrieved_flow["name"] == "My Test Flow"
    assert retrieved_flow["nodes"][0]["id"] == "n1"

    # Verify it was written to disk
    with open(TEST_FLOWS_FILE, "r") as f:
        disk_data = json.load(f)
    assert flow_id in disk_data

def test_list_flows_sorted(flow_manager):
    """Tests that flows are listed with the newest first."""
    flow1 = flow_manager.save_flow("Flow 1", [], [])
    import time; time.sleep(0.01) # ensure different timestamps
    flow2 = flow_manager.save_flow("Flow 2", [], [])
    
    flow_list = flow_manager.list_flows()
    assert len(flow_list) == 2
    assert flow_list[0]["id"] == flow2["id"] # Newest first
    assert flow_list[1]["id"] == flow1["id"]

def test_delete_flow(flow_manager):
    """Tests deleting an existing and non-existing flow."""
    flow_data = flow_manager.save_flow("To Be Deleted", [], [])
    flow_id = flow_data["id"]
    
    assert flow_manager.get_flow(flow_id) is not None
    
    result = flow_manager.delete_flow(flow_id)
    assert result is True
    assert flow_manager.get_flow(flow_id) is None
    
    # Test deleting non-existent
    result_non_existent = flow_manager.delete_flow("non-existent-id")
    assert result_non_existent is False