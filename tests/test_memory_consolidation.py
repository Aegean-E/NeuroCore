import pytest
import numpy as np
from unittest.mock import MagicMock, patch, AsyncMock
from modules.memory.consolidation import MemoryConsolidator

@pytest.fixture
def mock_store():
    with patch("modules.memory.consolidation.memory_store") as ms:
        # Mock executor to allow run_in_executor to work in tests if needed,
        # though we usually patch the method being run.
        ms.executor = None 
        yield ms

@pytest.fixture
def mock_llm():
    with patch("modules.memory.consolidation.LLMBridge") as MockBridge:
        yield MockBridge.return_value

@pytest.mark.asyncio
async def test_consolidation_logic(mock_store, mock_llm):
    """Test that high similarity + entailment triggers consolidation."""
    
    # Setup data: Mem 1 and Mem 2 are similar. Mem 3 is different.
    memories = [
        {"id": 1, "text": "Apple is red"},
        {"id": 2, "text": "Apples are red"},
        {"id": 3, "text": "Sky is blue"}
    ]
    
    # Mock Similarity matrix (3x3)
    # 1-2: High (0.95)
    # 1-3: Low (0.0)
    sim_matrix = np.array([
        [1.0, 0.95, 0.0],
        [0.95, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ])
    
    # Patch the blocking method that fetches data
    with patch.object(MemoryConsolidator, '_fetch_and_process_candidates', return_value=(memories, sim_matrix)):
        consolidator = MemoryConsolidator()
        consolidator.llm = mock_llm
        
        # Mock LLM entailment check to return YES
        mock_llm.chat_completion = AsyncMock(return_value={
            "choices": [{"message": {"content": "YES"}}]
        })
        
        count = await consolidator.run()
        
        assert count == 1
        # Should consolidate 1 into 2 (since 1 comes first in loop, it finds 2, and sets 1 as child of 2)
        mock_store.set_parent.assert_called_with(child_id=1, parent_id=2)

@pytest.mark.asyncio
async def test_consolidation_entailment_check(mock_llm):
    """Test the LLM entailment verification logic."""
    consolidator = MemoryConsolidator()
    consolidator.llm = mock_llm
    
    # Test YES
    mock_llm.chat_completion = AsyncMock(return_value={"choices": [{"message": {"content": "YES"}}]})
    assert await consolidator._check_entailment("A", "B") is True
    
    # Test NO
    mock_llm.chat_completion = AsyncMock(return_value={"choices": [{"message": {"content": "NO"}}]})
    assert await consolidator._check_entailment("A", "B") is False