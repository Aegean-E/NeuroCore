import os
import pytest
import json
import numpy as np
from unittest.mock import MagicMock, patch, AsyncMock
from fastapi.testclient import TestClient
from main import app
from modules.knowledge_base.backend import FaissDocumentStore
from modules.knowledge_base.node import KnowledgeQueryExecutor

TEST_DB_PATH = "test_kb.sqlite3"

@pytest.fixture
def kb_store():
    if os.path.exists(TEST_DB_PATH): os.remove(TEST_DB_PATH)
    store = FaissDocumentStore(db_path=TEST_DB_PATH)
    yield store
    if os.path.exists(TEST_DB_PATH): os.remove(TEST_DB_PATH)
    if os.path.exists(TEST_DB_PATH.replace(".sqlite3", ".faiss")): 
        os.remove(TEST_DB_PATH.replace(".sqlite3", ".faiss"))

def test_kb_backend_add_search(kb_store):
    """Test adding documents and searching them."""
    # Add document
    chunks = [
        {"text": "Python is a language", "embedding": np.array([1.0] + [0.0]*767, dtype='float32'), "page_number": 1},
        {"text": "Java is verbose", "embedding": np.array([0.0] + [1.0]*767, dtype='float32'), "page_number": 2}
    ]
    
    kb_store.add_document(
        file_hash="hash1", 
        filename="test.pdf", 
        file_type="pdf", 
        file_size=100, 
        page_count=2, 
        chunks=chunks
    )
    
    assert kb_store.get_total_documents() == 1
    assert kb_store.get_total_chunks() == 2
    
    # Search (Exact match for Python)
    results = kb_store.search([1.0] + [0.0]*767, limit=1)
    assert len(results) == 1
    assert results[0]["content"] == "Python is a language"
    assert results[0]["source"] == "test.pdf"

@pytest.mark.asyncio
async def test_kb_node_execution():
    """Test the KnowledgeQueryExecutor."""
    
    # Mock Backend Search
    with patch("modules.knowledge_base.node.document_store") as mock_store:
        executor = KnowledgeQueryExecutor()
        # Mock LLM embedding
        executor.llm.get_embedding = AsyncMock(return_value=[0.1, 0.2])

        # Mock get_total_documents to return non-zero so the query proceeds
        mock_store.get_total_documents.return_value = 5
        mock_store.search.return_value = [
            {"content": "Relevant info", "source": "doc.pdf", "page": 1}
        ]
        mock_store.search_hybrid.return_value = [
            {"content": "Relevant info", "source": "doc.pdf", "page": 1}
        ]
        
        input_data = {"messages": [{"role": "user", "content": "Query"}]}
        result = await executor.receive(input_data, config={"limit": 2})
        
        # Verify embedding call
        executor.llm.get_embedding.assert_called_with("Query")
        
        # Verify search call
        # The node might call search OR search_hybrid depending on config/implementation
        if mock_store.search.called:
            args, kwargs = mock_store.search.call_args
            assert args[0] == [0.1, 0.2]
            assert kwargs.get('limit') == 2 or (len(args) > 1 and args[1] == 2)
        else:
            assert mock_store.search_hybrid.called
            # search_hybrid also takes query_embedding
        
        # Verify output injection
        assert "knowledge_context" in result
        assert "Relevant info" in result["knowledge_context"]
        # Note: The KnowledgeQueryExecutor returns knowledge_context as a separate field,
        # not injected into messages. This is by design to let System Prompt handle injection.

@pytest.fixture
def client():
    with TestClient(app) as c:
        c.app.state.module_manager.enable_module("knowledge_base")
        yield c

def test_kb_router_list(client):
    """Test listing documents."""
    with patch("modules.knowledge_base.router.document_store") as mock_store:
        mock_store.list_documents.return_value = [
            {
                "id": 1, "filename": "test.pdf", "file_type": "pdf", 
                "page_count": 10, "chunk_count": 5, "created_at": 1234567890
            }
        ]
        
        response = client.get("/knowledge_base/list")
        assert response.status_code == 200
        assert "test.pdf" in response.text
        assert "5 chunks" in response.text

def test_kb_router_delete(client):
    """Test deleting a document."""
    with patch("modules.knowledge_base.router.document_store") as mock_store, \
         patch("os.remove") as mock_remove, \
         patch("os.path.exists", return_value=True):
        
        # Mock list to find filename
        # Note: delete_doc uses get_document, not list_documents, but we mock list_documents 
        # because the route returns the updated list at the end.
        mock_store.get_document.return_value = {"id": 1, "filename": "test.pdf"}
        mock_store.list_documents.return_value = []
        
        response = client.delete("/knowledge_base/delete/1")
        assert response.status_code == 200
        
        # Verify DB delete
        mock_store.delete_document.assert_called_with(1)
        
        # Verify file delete (uploaded file + processed json)
        assert mock_remove.call_count >= 2 

def test_kb_router_upload_flow(client):
    """Test the upload flow (mocking heavy processing)."""
    with patch("modules.knowledge_base.router.DocumentProcessor") as MockProcessor, \
         patch("modules.knowledge_base.router.document_store") as mock_store, \
         patch("builtins.open", new_callable=MagicMock), \
         patch("shutil.copyfileobj"), \
         patch("os.path.getsize", return_value=1024), \
         patch("json.dump"), \
         patch("modules.knowledge_base.router.templates.env.get_template") as mock_get_template:
        
        mock_tmpl = MagicMock()
        mock_tmpl.render.return_value = "<div>Progress</div>"
        mock_get_template.return_value = mock_tmpl
        
        # Mock Processor
        proc_instance = MockProcessor.return_value
        proc_instance.process_document = AsyncMock(return_value=(
            [{"text": "chunk", "embedding": [0.1]}], # chunks
            5, # page count
            "pdf" # type
        ))
        
        # Mock Store check
        mock_store.compute_file_hash.return_value = "hash123"
        mock_store.document_exists.return_value = False
        
        files = {"files": ("test.pdf", b"content", "application/pdf")}
        response = client.post("/knowledge_base/upload", files=files)
        
        assert response.status_code == 200
        mock_store.add_document.assert_called()

# ---------------------------------------------------------------------------
# FAISS / SQLite split-brain regression tests
# ---------------------------------------------------------------------------

class TestFaissDimensionMismatchWarning:
    """Regression tests for the FAISS/SQLite dimension-mismatch split-brain bug.

    When an embedding dimension mismatches the existing FAISS index, chunks are
    already committed to SQLite. _add_embeddings_to_faiss() must return a
    non-zero skip count, and add_document() must log a WARNING so operators
    know those chunks are keyword-searchable only.
    """

    @pytest.fixture
    def store(self, tmp_path):
        db_path = str(tmp_path / "kb.sqlite3")
        s = FaissDocumentStore(db_path=db_path)
        yield s

    def _make_chunk(self, dim: int, value: float = 1.0):
        emb = np.zeros(dim, dtype="float32")
        emb[0] = value
        return {"text": f"chunk dim={dim}", "embedding": emb, "page_number": 1}

    def test_skip_count_returned_on_dimension_mismatch(self, store):
        """_add_embeddings_to_faiss() returns the number of skipped embeddings."""
        import faiss

        # Seed the index with dimension=8
        dim_a = 8
        store.faiss_index = faiss.IndexIDMap(faiss.IndexFlatIP(dim_a))
        emb_a = np.ones((1, dim_a), dtype="float32")
        store.faiss_index.add_with_ids(emb_a, np.array([1], dtype="int64"))
        assert store.faiss_index.ntotal == 1

        # Now try to add an embedding with dimension=4 (mismatch)
        dim_b = 4
        emb_b = np.ones(dim_b, dtype="float32")
        skipped = store._add_embeddings_to_faiss([emb_b], [99], save_index=False)

        assert skipped == 1, f"Expected 1 skipped, got {skipped}"
        assert store.faiss_index.ntotal == 1  # index unchanged

    def test_add_document_logs_warning_on_dimension_mismatch(self, store, caplog):
        """add_document() logs a WARNING when chunks are skipped by FAISS."""
        import logging
        import faiss

        # Pre-populate the FAISS index with dimension=8
        dim_a = 8
        store.faiss_index = faiss.IndexIDMap(faiss.IndexFlatIP(dim_a))
        emb_seed = np.ones((1, dim_a), dtype="float32")
        store.faiss_index.add_with_ids(emb_seed, np.array([1], dtype="int64"))

        # add_document with wrong-dimension embedding
        dim_b = 4
        chunks = [self._make_chunk(dim_b)]

        with caplog.at_level(logging.WARNING):
            store.add_document(
                file_hash="aabbcc",
                filename="test.txt",
                file_type="txt",
                file_size=100,
                page_count=1,
                chunks=chunks,
                upload_source="unit-test",
            )

        assert any(
            "keyword search only" in record.message or "dimension mismatch" in record.message.lower()
            for record in caplog.records
        ), f"Expected a dimension-mismatch warning, got: {[r.message for r in caplog.records]}"

    def test_no_warning_when_all_chunks_indexed(self, store, caplog):
        """add_document() must NOT warn when all chunks are accepted by FAISS."""
        import logging

        dim = 8
        chunks = [self._make_chunk(dim)]

        with caplog.at_level(logging.WARNING):
            store.add_document(
                file_hash="ddeeff",
                filename="ok.txt",
                file_type="txt",
                file_size=50,
                page_count=1,
                chunks=chunks,
                upload_source="unit-test",
            )

        dimension_warnings = [
            r for r in caplog.records
            if "keyword search only" in r.message
        ]
        assert dimension_warnings == [], (
            f"Unexpected warnings: {[w.message for w in dimension_warnings]}"
        )
