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

    # Setup: mem 1 (older) and mem 2 (newer) are similar; mem 3 is unrelated.
    # created_at uses Unix timestamps; lower value = older.
    memories = [
        {"id": 1, "text": "Apple is red",   "created_at": 1000},
        {"id": 2, "text": "Apples are red", "created_at": 2000},
        {"id": 3, "text": "Sky is blue",    "created_at": 3000},
    ]

    # _find_similar_pairs_faiss returns (id_a, id_b, similarity) tuples.
    similar_pairs = [(1, 2, 0.95)]

    consolidator = MemoryConsolidator()
    consolidator.llm = mock_llm

    # Mock LLM entailment check to return YES
    mock_llm.chat_completion = AsyncMock(return_value={
        "choices": [{"message": {"content": "YES"}}]
    })

    with patch.object(MemoryConsolidator, '_fetch_candidates_with_timestamps',
                      return_value=memories), \
         patch.object(MemoryConsolidator, '_find_similar_pairs_faiss',
                      return_value=similar_pairs), \
         patch.object(MemoryConsolidator, '_get_processed_pairs', return_value=set()), \
         patch.object(MemoryConsolidator, '_log_consolidation'):

        count = await consolidator.run()

    assert count == 1
    # mem 1 is older (created_at=1000 < 2000), so it becomes the child of mem 2.
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