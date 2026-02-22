import pytest
import asyncio
import numpy as np
from unittest.mock import MagicMock, patch, AsyncMock
from modules.knowledge_base.processor import DocumentProcessor
from modules.knowledge_base.backend import FaissDocumentStore
from core.llm import LLMBridge

@pytest.mark.asyncio
async def test_processor_parallel_embeddings():
    """Test that embeddings are generated in parallel and connection is reused."""
    
    # Mock LLMBridge
    mock_bridge = MagicMock(spec=LLMBridge)
    mock_bridge.base_url = "http://test"
    mock_bridge.api_key = "key"
    mock_bridge.embedding_base_url = "http://test"
    mock_bridge.embedding_model = "model"
    
    # We need to mock the internal batch_bridge created inside _generate_embeddings
    with patch("modules.knowledge_base.processor.httpx.AsyncClient") as MockClient, \
         patch("modules.knowledge_base.processor.LLMBridge") as MockBridgeClass:
        
        mock_client_instance = MockClient.return_value.__aenter__.return_value
        mock_batch_bridge = MockBridgeClass.return_value
        mock_batch_bridge.get_embedding = AsyncMock(side_effect=lambda text: [0.1] * 10)
        
        processor = DocumentProcessor(mock_bridge)
        chunks = [{"text": f"chunk {i}"} for i in range(10)]
        
        await processor._generate_embeddings(chunks)
        
        # Verify client was created (context manager used)
        MockClient.assert_called_once()
        
        # Verify embeddings were populated
        assert len(chunks) == 10
        assert chunks[0]["embedding"] == [0.1] * 10
        
        # Verify calls were made
        assert mock_batch_bridge.get_embedding.call_count == 10

def test_backend_optimized_search():
    """Test the optimized SQL IN search."""
    # Setup temporary store
    import os
    db_path = "test_kb_opt.sqlite3"
    if os.path.exists(db_path): os.remove(db_path)
    
    store = FaissDocumentStore(db_path=db_path)
    
    try:
        # Add dummy data
        chunks = [
            {"text": "A", "embedding": np.array([1.0, 0.0], dtype='float32')},
            {"text": "B", "embedding": np.array([0.0, 1.0], dtype='float32')},
            {"text": "C", "embedding": np.array([0.5, 0.5], dtype='float32')}
        ]
        store.add_document("hash", "test.txt", "txt", 100, 1, chunks)
        
        # Mock FAISS search to return specific indices and scores
        # Let's say it returns index 3 (C) then 2 (B)
        # Note: IDs in DB start at 1. Chunks are inserted sequentially.
        # Chunk A -> ID 1, B -> ID 2, C -> ID 3.
        # FAISS IDs match DB IDs.
        
        # We mock the internal faiss_index.search
        store.faiss_index.search = MagicMock(return_value=(
            np.array([[0.9, 0.8]], dtype='float32'), # Scores
            np.array([[3, 2]], dtype='int64')        # Indices (C, B)
        ))
        store.faiss_index.ntotal = 3
        
        results = store.search([0.5, 0.5], limit=2)
        
        assert len(results) == 2
        # Should be sorted by score
        assert results[0]["content"] == "C"
        assert results[0]["score"] == pytest.approx(0.9)
        assert results[1]["content"] == "B"
        assert results[1]["score"] == pytest.approx(0.8)
        
    finally:
        # Ensure store releases resources
        if 'store' in locals():
            store.faiss_index = None
            del store
        if os.path.exists(db_path): os.remove(db_path)
        if os.path.exists(db_path.replace(".sqlite3", ".faiss")): 
            os.remove(db_path.replace(".sqlite3", ".faiss"))