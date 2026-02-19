import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from core.flow_runner import FlowRunner

@pytest.fixture
def mock_flow():
    """A fixture for a valid, simple, linear flow."""
    return {
        "id": "test-flow",
        "nodes": [
            {"id": "node-1", "moduleId": "module_a", "nodeTypeId": "type_a", "name": "Start"},
            {"id": "node-2", "moduleId": "module_b", "nodeTypeId": "type_b", "name": "Middle"},
            {"id": "node-3", "moduleId": "module_c", "nodeTypeId": "type_c", "name": "End"},
        ],
        "connections": [
            {"from": "node-1", "to": "node-2"},
            {"from": "node-2", "to": "node-3"},
        ]
    }

@pytest.fixture
def mock_flow_with_cycle():
    """A fixture for a flow with a cycle, which should be invalid."""
    return {
        "id": "cycle-flow",
        "nodes": [
            {"id": "node-1", "moduleId": "module_a", "nodeTypeId": "type_a", "name": "A"},
            {"id": "node-2", "moduleId": "module_b", "nodeTypeId": "type_b", "name": "B"},
        ],
        "connections": [
            {"from": "node-1", "to": "node-2"},
            {"from": "node-2", "to": "node-1"},
        ]
    }

def test_topological_sort_success(mock_flow):
    """Tests that a valid flow produces the correct execution order."""
    with patch('core.flow_runner.flow_manager') as mock_fm:
        mock_fm.get_flow.return_value = mock_flow
        runner = FlowRunner(flow_id="test-flow")
        assert runner.execution_order == ["node-1", "node-2", "node-3"]

def test_topological_sort_cycle_detection(mock_flow_with_cycle):
    """Tests that a flow with a cycle raises an exception."""
    with patch('core.flow_runner.flow_manager') as mock_fm:
        mock_fm.get_flow.return_value = mock_flow_with_cycle
        with pytest.raises(Exception, match="Flow contains a cycle"):
            FlowRunner(flow_id="cycle-flow")

@pytest.mark.asyncio
async def test_flow_run_success(mock_flow):
    """Tests the successful execution of a flow from start to finish."""
    with patch('core.flow_runner.flow_manager') as mock_fm, \
         patch('importlib.import_module') as mock_import:
        
        mock_fm.get_flow.return_value = mock_flow

        # Mock the executor classes and their methods to be simple pass-throughs
        mock_executor_instance = MagicMock()
        mock_executor_instance.receive = AsyncMock(side_effect=lambda d: d)
        mock_executor_instance.send = AsyncMock(side_effect=lambda d: d)
        mock_executor_class = MagicMock(return_value=mock_executor_instance)
        mock_node_dispatcher = MagicMock()
        mock_node_dispatcher.get_executor_class = AsyncMock(return_value=mock_executor_class)
        mock_import.return_value = mock_node_dispatcher

        runner = FlowRunner(flow_id="test-flow")
        initial_data = {"data": "start"}
        result = await runner.run(initial_data)

        assert result == initial_data
        assert mock_executor_instance.receive.call_count == 3
        assert mock_executor_instance.send.call_count == 3

@pytest.mark.asyncio
async def test_flow_run_node_execution_error(mock_flow):
    """Tests that an error during a node's execution is caught and reported."""
    with patch('core.flow_runner.flow_manager') as mock_fm, \
         patch('importlib.import_module') as mock_import:
        
        mock_fm.get_flow.return_value = mock_flow

        # Mock an executor that raises an error on the second node
        async def receive_side_effect(data):
            if mock_executor_instance.receive.call_count == 2: # Fail on second call
                raise ValueError("Something went wrong")
            return data

        mock_executor_instance = MagicMock(receive=AsyncMock(side_effect=receive_side_effect), send=AsyncMock(side_effect=lambda d: d))
        mock_import.return_value = MagicMock(get_executor_class=AsyncMock(return_value=MagicMock(return_value=mock_executor_instance)))

        runner = FlowRunner(flow_id="test-flow")
        result = await runner.run({"data": "start"})

        assert "error" in result
        assert "Execution failed at node 'Middle': Something went wrong" in result["error"]