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

        mock_store.search.return_value = [
            {"content": "Relevant info", "source": "doc.pdf", "page": 1}
        ]
        
        input_data = {"messages": [{"role": "user", "content": "Query"}]}
        result = await executor.receive(input_data, config={"limit": 2})
        
        # Verify embedding call
        executor.llm.get_embedding.assert_called_with("Query")
        
        # Verify search call
        mock_store.search.assert_called_with([0.1, 0.2], 2)
        
        # Verify output injection
        assert "knowledge_context" in result
        assert "Relevant info" in result["knowledge_context"]
        assert "Relevant Knowledge Base Context" in result["messages"][-1]["content"]

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