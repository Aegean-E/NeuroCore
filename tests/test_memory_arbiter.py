import pytest
from unittest.mock import MagicMock
from modules.memory.arbiter import MemoryArbiter

@pytest.fixture
def mock_store():
    store = MagicMock()
    # Default behavior: add_entry returns a dummy ID
    store.add_entry.return_value = 123
    # Default identity
    store.compute_identity.return_value = "hash"
    return store

def test_arbiter_consider_success(mock_store):
    """Test that valid high-confidence memories are passed to the store."""
    arbiter = MemoryArbiter(mock_store)
    mid = arbiter.consider(text="Test fact", mem_type="FACT", confidence=0.8)
    
    assert mid == 123
    mock_store.add_entry.assert_called_once()
    args, kwargs = mock_store.add_entry.call_args
    assert kwargs["text"] == "Test fact"
    assert kwargs["mem_type"] == "FACT"

def test_arbiter_confidence_gate(mock_store):
    """Test that low-confidence memories are rejected."""
    arbiter = MemoryArbiter(mock_store)
    # FACT requires 0.7 by default
    mid = arbiter.consider(text="Low confidence", mem_type="FACT", confidence=0.5)
    
    assert mid is None
    mock_store.add_entry.assert_not_called()

def test_arbiter_unknown_type_defaults_to_fact(mock_store):
    """Test that unknown memory types default to FACT."""
    arbiter = MemoryArbiter(mock_store)
    mid = arbiter.consider(text="Unknown type", mem_type="UNKNOWN", confidence=0.9)
    
    assert mid == 123
    args, kwargs = mock_store.add_entry.call_args
    assert kwargs["mem_type"] == "FACT"

def test_arbiter_backend_rejection(mock_store):
    """Test handling of backend rejection (e.g. duplicates)."""
    arbiter = MemoryArbiter(mock_store)
    mock_store.add_entry.return_value = -1 # Simulate duplicate
    
    mid = arbiter.consider(text="Duplicate", mem_type="FACT", confidence=1.0)
    assert mid is None

def test_arbiter_consider_batch(mock_store):
    """Test batch processing of memories."""
    arbiter = MemoryArbiter(mock_store)
    
    candidates = [
        {"text": "Mem 1", "type": "FACT"},
        {"text": "Mem 2", "type": "RULE", "confidence": 0.95},
        {"text": "Mem 3", "type": "FACT", "confidence": 0.1} # Should be rejected
    ]
    
    # Mock add_entry to return IDs
    mock_store.add_entry.side_effect = [101, 102] 
    
    ids = arbiter.consider_batch(candidates)
    
    assert len(ids) == 2
    assert ids == [101, 102]
    assert mock_store.add_entry.call_count == 2

def test_arbiter_passes_embedding(mock_store):
    """Test that embedding is passed to the store."""
    arbiter = MemoryArbiter(mock_store)
    embedding = [0.1, 0.2, 0.3]
    mid = arbiter.consider(text="Embedded", mem_type="FACT", confidence=1.0, embedding=embedding)
    
    assert mid == 123
    args, kwargs = mock_store.add_entry.call_args
    assert kwargs["embedding"] == embedding