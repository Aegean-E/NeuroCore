import os
import pytest
import json
from unittest.mock import patch
import numpy as np
from modules.memory.backend import MemoryStore

TEST_DB_PATH = "test_memory.sqlite3"

@pytest.fixture
def store():
    # Setup
    if os.path.exists(TEST_DB_PATH):
        os.remove(TEST_DB_PATH)
    
    # Initialize store with test DB
    ms = MemoryStore(db_path=TEST_DB_PATH)
    yield ms
    
    # Teardown
    if os.path.exists(TEST_DB_PATH):
        os.remove(TEST_DB_PATH)
    if os.path.exists(TEST_DB_PATH.replace(".sqlite3", ".faiss")):
        os.remove(TEST_DB_PATH.replace(".sqlite3", ".faiss"))

def test_add_and_get_entry(store):
    """Test adding a memory and retrieving it via recent list."""
    mid = store.add_entry(text="The sky is blue", confidence=0.9, subject="User", mem_type="FACT")
    assert mid != -1
    
    recent = store.get_recent(limit=1)
    assert len(recent) == 1
    assert recent[0]["text"] == "The sky is blue"
    assert recent[0]["id"] == mid

def test_duplicate_prevention(store):
    """Test that identical text (identity hash) is not added twice."""
    mid1 = store.add_entry(text="Unique thought", confidence=1.0)
    mid2 = store.add_entry(text="Unique thought", confidence=1.0)
    
    assert mid1 != -1
    assert mid2 == -1  # Should be rejected as duplicate

def test_search_linear_fallback(store):
    """Test vector search (linear fallback if FAISS not active/mocked)."""
    # Add a few entries
    # We use dummy embeddings: [1, 0] vs [0, 1]
    store.add_entry(text="Apple", embedding=[1.0, 0.0])
    store.add_entry(text="Banana", embedding=[0.0, 1.0])
    
    # Search close to Apple
    results = store.search(query_embedding=[0.9, 0.1], limit=1)
    assert len(results) == 1
    assert results[0]["text"] == "Apple"

def test_delete_entry(store):
    mid = store.add_entry(text="To delete", embedding=[0.1, 0.2])
    assert len(store.get_recent()) == 1
    
    # Verify it's searchable
    assert len(store.search([0.1, 0.2])) == 1
    
    store.delete_entry(mid)
    assert len(store.get_recent()) == 0
    # Verify it's gone from search (FAISS/Linear)
    assert len(store.search([0.1, 0.2])) == 0

def test_browse_filtering(store):
    store.add_entry(text="Fact 1", mem_type="FACT")
    store.add_entry(text="Rule 1", mem_type="RULE")
    
    # Filter by type
    facts = store.browse(mem_type="FACT")
    assert len(facts) == 1
    assert facts[0]["text"] == "Fact 1"
    
    # Search text
    results = store.browse(search_text="Rule")
    assert len(results) == 1
    assert results[0]["text"] == "Rule 1"

def test_stats(store):
    store.add_entry(text="A", subject="User")
    stats = store.get_memory_stats()
    assert stats["total"] == 1
    assert stats["user"] == 1

def test_embedding_parsing_robustness(store):
    """Test that the store can parse both binary and JSON string embeddings."""
    # 1. Add entry with binary embedding (standard)
    mid1 = store.add_entry(text="Binary", embedding=[0.1, 0.2])
    
    # 2. Manually insert a JSON string embedding (simulating legacy/external data)
    with store._connect() as con:
        con.execute("INSERT INTO memories (identity, type, text, confidence, created_at, embedding) VALUES (?, ?, ?, ?, ?, ?)",
                    ("hash_json", "FACT", "JSON", 1.0, 12345, json.dumps([0.1, 0.2])))
    
    # Force linear search by temporarily hiding the FAISS index
    # This ensures we are testing the _parse_embedding logic in the linear fallback
    with patch.object(store, 'faiss_index', None):
        results = store.search(query_embedding=[0.1, 0.2], limit=2)
        assert len(results) == 2

def test_faiss_sync_recovery():
    """Test that the FAISS index is rebuilt from DB if missing."""
    db_path = "test_memory_sync.sqlite3"
    index_path = "test_memory_sync.faiss"
    
    # Cleanup start
    if os.path.exists(db_path): os.remove(db_path)
    if os.path.exists(index_path): os.remove(index_path)

    try:
        # 1. Create store and add data
        ms = MemoryStore(db_path=db_path)
        ms.add_entry("Test Sync", embedding=[0.5, 0.5])
        del ms # Close/Release lock

        # 2. Delete index file to simulate loss/corruption
        if os.path.exists(index_path):
            os.remove(index_path)
        
        # 3. Re-open store (should trigger rebuild)
        ms2 = MemoryStore(db_path=db_path)
        
        # 4. Check if index is working
        results = ms2.search([0.5, 0.5], limit=1)
        assert len(results) == 1
        assert results[0]["text"] == "Test Sync"
        
    finally:
        if os.path.exists(db_path): os.remove(db_path)
        if os.path.exists(index_path): os.remove(index_path)

def test_browse_pagination(store):
    """Test pagination in browse method."""
    # Add 10 entries
    for i in range(10):
        store.add_entry(text=f"Entry {i}", created_at=1000+i)
        
    # Page 1: Top 5 (newest first)
    page1 = store.browse(limit=5, offset=0)
    assert len(page1) == 5
    assert page1[0]["text"] == "Entry 9"
    
    # Page 2: Next 5
    page2 = store.browse(limit=5, offset=5)
    assert len(page2) == 5
    assert page2[0]["text"] == "Entry 4"

def test_manual_faiss_sync(store):
    """Test that _sync_faiss_index rebuilds index if counts mismatch."""
    if not store.faiss_index:
        pytest.skip("FAISS not available")
        
    store.add_entry("Test", embedding=[0.1, 0.1])
    store.faiss_index.reset() # Force desync
    assert store.faiss_index.ntotal == 0
    
    store._sync_faiss_index()
    assert store.faiss_index.ntotal == 1

def test_embedding_parsing_invalid(store):
    """Test that invalid embedding blobs return None."""
    # Invalid JSON
    assert store._parse_embedding('{"invalid": json') is None
    
    # None
    assert store._parse_embedding(None) is None