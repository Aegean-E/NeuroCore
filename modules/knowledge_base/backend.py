import os
import logging
import sqlite3
import time
import hashlib
import json
import threading
from typing import List, Dict, Optional, Tuple
import numpy as np
from core.settings import settings

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    logging.warning("⚠️ FAISS not installed. Install with: pip install faiss-cpu")
    FAISS_AVAILABLE = False


class FaissDocumentStore:
    """
    FAISS-enhanced document store with fast vector search.
    
    Uses:
    - SQLite for metadata (filenames, page numbers, etc.)
    - FAISS for vector embeddings (fast similarity search)
    """

    def __init__(self, db_path: str = "data/knowledge_base.sqlite3", embed_fn=None):
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self.db_path = db_path
        self.embed_fn = embed_fn
        self.faiss_index = None
        self.index_lock = threading.Lock()
        self._init_db()
        
        # Load FAISS configuration from settings
        self.faiss_index_type = settings.get("faiss_index_type", "IndexFlatIP")
        self.faiss_nlist = settings.get("faiss_nlist", 100) # For IndexIVFFlat
        self.faiss_nprobe = settings.get("faiss_nprobe", 10)

        self._load_faiss_index()
        self._sync_faiss_index()
        self._sync_fts_index()

    # --------------------------
    # Database Initialization
    # --------------------------

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.db_path)
        con.execute("PRAGMA journal_mode=WAL;")
        return con

    def _init_db(self) -> None:
        with self._connect() as con:
            # Documents table (metadata)
            con.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_hash TEXT NOT NULL UNIQUE,
                    filename TEXT NOT NULL,
                    file_type TEXT NOT NULL,
                    file_size INTEGER NOT NULL,
                    page_count INTEGER,
                    chunk_count INTEGER NOT NULL,
                    upload_source TEXT NOT NULL,
                    created_at INTEGER NOT NULL
                )
            """)

            # Chunks table (text segments with metadata)
            con.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id INTEGER NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    page_number INTEGER,
                    created_at INTEGER NOT NULL,
                    embedding TEXT,
                    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
                )
            """)

            # Create indexes
            con.execute("CREATE INDEX IF NOT EXISTS idx_file_hash ON documents(file_hash)")
            con.execute("CREATE INDEX IF NOT EXISTS idx_document_id ON chunks(document_id)")
            
            # Migration: Add embedding column if it doesn't exist
            try:
                con.execute("ALTER TABLE chunks ADD COLUMN embedding TEXT")
            except sqlite3.OperationalError:
                pass

            # 1. Enable FTS5 Virtual Table for Keyword Search
            con.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts 
                USING fts5(text, content='chunks', content_rowid='id');
            """)
            
            # 2. Trigger to keep FTS synced with Chunks
            con.execute("""
                CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
                  INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
                END;
            """)
            con.execute("""
                CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
                  INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES('delete', old.id, old.text);
                END;
            """)

    # --------------------------
    # FAISS Integration
    # --------------------------

    def _load_faiss_index(self):
        """Load or create FAISS index"""
        index_path = self.db_path.replace(".sqlite3", ".faiss")
        
        if os.path.exists(index_path):
            try:
                with self.index_lock:
                    # Load existing index
                    self.faiss_index = faiss.read_index(index_path)
                    
                    # Sanity check: Verify dimensions and searchability
                    if self.faiss_index and self.faiss_index.ntotal > 0:
                        # Perform a dummy search to ensure index is usable
                        d = self.faiss_index.d
                        dummy = np.random.rand(1, d).astype('float32')
                        faiss.normalize_L2(dummy)
                        self.faiss_index.search(dummy, 1)
            except Exception as e:
                logging.error(f"⚠️ FAISS index file is corrupted or incompatible: {e}")
                logging.info("🔧 Deleting corrupted index and creating a fresh one...")
                try:
                    os.remove(index_path)
                except:
                    pass
                self._create_empty_index()
        else:
            self._create_empty_index()
            
    def _create_empty_index(self):
        """Helper to create a new empty index"""
        dimension = self._detect_embedding_dimension()
        with self.index_lock:
            logging.info(f"🔧 Creating FAISS index with dimension: {dimension}")
            if self.faiss_index_type == "IndexIVFFlat":
                # For IndexIVFFlat, we need to train the index
                quantizer = faiss.IndexFlatIP(dimension)
                self.faiss_index = faiss.IndexIDMap(faiss.IndexIVFFlat(quantizer, dimension, self.faiss_nlist, faiss.METRIC_INNER_PRODUCT))
                # Training will happen when the first batch of embeddings is added
            else:
                # Default to IndexFlatIP
                self.faiss_index = faiss.IndexIDMap(faiss.IndexFlatIP(dimension))
    
    def _detect_embedding_dimension(self):
        """Detect the embedding dimension from the model"""
        # Try to get dimension from existing chunks
        try:
            with self._connect() as con:
                sample_chunk = con.execute("""
                    SELECT embedding FROM chunks WHERE embedding IS NOT NULL LIMIT 1
                """).fetchone()
                
                if sample_chunk and sample_chunk[0]:
                    emb = json.loads(sample_chunk[0])
                    return len(emb)
        except Exception:
            pass
        
        # Default fallback
        return 768 

    def _save_faiss_index(self):
        """Save FAISS index to disk"""
        index_path = self.db_path.replace(".sqlite3", ".faiss")
        temp_path = index_path + ".tmp"
        try:
            with self.index_lock:
                if self.faiss_index:
                    faiss.write_index(self.faiss_index, temp_path)
                    if os.path.exists(index_path):
                        os.remove(index_path)
                    os.rename(temp_path, index_path)
        except Exception as e:
            logging.error(f"⚠️ Failed to save FAISS index: {e}")
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def _add_embeddings_to_faiss(self, embeddings: List[np.ndarray], chunk_ids: List[int], save_index: bool = True):
        """Add embeddings to FAISS index"""
        if not embeddings:
            return
            
        # Defensive check for inhomogeneous shapes
        valid_embeddings = []
        valid_ids = []
        expected_dim = self.faiss_index.d if self.faiss_index else None
        
        for emb, cid in zip(embeddings, chunk_ids):
            if expected_dim is None or len(emb) == expected_dim:
                # Ensure embedding is finite to prevent index corruption
                if np.all(np.isfinite(emb)) and not np.all(emb == 0):
                    valid_embeddings.append(emb)
                    valid_ids.append(cid)
        
        if not valid_embeddings: return
        embeddings_array = np.array(valid_embeddings).astype('float32')
        
        # Normalize for inner product search
        faiss.normalize_L2(embeddings_array)
        
        with self.index_lock:
            # Train IndexIVFFlat if not already trained
            if self.faiss_index_type == "IndexIVFFlat" and not self.faiss_index.is_trained:
                logging.info(f"🔧 Training FAISS IndexIVFFlat with {len(embeddings)} vectors (nlist={self.faiss_nlist})...")
                # Ensure enough data for training
                if len(embeddings_array) >= self.faiss_nlist:
                    self.faiss_index.train(embeddings_array)
            if self.faiss_index:
                # Add to index
                ids_array = np.array(valid_ids).astype('int64')
                self.faiss_index.add_with_ids(embeddings_array, ids_array)
        
        # Save index
        if save_index:
            self._save_faiss_index()

    def _sync_faiss_index(self):
        """Ensure FAISS index is in sync with SQLite chunks."""
        try:
            with self._connect() as con:
                count = con.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            
            # If DB has data but FAISS is empty/None or mismatched, rebuild
            with self.index_lock:
                index_count = self.faiss_index.ntotal if self.faiss_index else 0
            
            if count > 0 and (index_count == 0 or index_count != count):
                logging.warning(f"⚠️ FAISS index out of sync (Index: {index_count}, DB: {count}). Rebuilding...")
                self._rebuild_index()
        except Exception as e:
            logging.error(f"⚠️ Error syncing FAISS index: {e}")

    def _sync_fts_index(self):
        """Ensure FTS index is populated."""
        try:
            with self._connect() as con:
                # Check if FTS is empty but chunks are not
                fts_count = con.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0]
                chunks_count = con.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
                
                if chunks_count > 0 and fts_count == 0:
                    logging.info("🔧 Populating FTS5 index...")
                    con.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')")
                    con.commit()
        except Exception as e:
            logging.warning(f"⚠️ FTS sync failed: {e}")

    def _rebuild_index(self):
        """Rebuild FAISS index from SQLite chunks."""
        # Reset index first
        with self.index_lock:
            self.faiss_index = None 

        with self._connect() as con:
            cur = con.execute("SELECT id, text, embedding FROM chunks ORDER BY id")
        
            batch_size = 1000
            total_processed = 0

            logging.info(f"🔄 Rebuilding index (batch size: {batch_size})...")
            
            while True:
                rows = cur.fetchmany(batch_size)
                if not rows:
                    break
                
                batch_embeddings = []
                batch_ids = []

                for r in rows:
                    chunk_id = r[0]
                    emb_json = r[2]
                    
                    emb = None
                    if emb_json:
                        try:
                            emb = np.array(json.loads(emb_json), dtype='float32')
                        except Exception:
                            pass
                    
                    if emb is not None:
                        batch_embeddings.append(emb)
                        batch_ids.append(chunk_id)
                
                # Initialize index on first batch if needed
                with self.index_lock:
                    if self.faiss_index is None and batch_embeddings:
                        dimension = len(batch_embeddings[0])
                        self.faiss_index = faiss.IndexIDMap(faiss.IndexFlatIP(dimension))

                # Add batch to FAISS immediately to free memory
                if batch_embeddings and self.faiss_index:
                    self._add_embeddings_to_faiss(batch_embeddings, batch_ids, save_index=False)

                total_processed += len(rows)
            
            # Ensure index exists even if DB was empty
            with self.index_lock:
                if self.faiss_index is None:
                    pass
            if not self.faiss_index:
                self._create_empty_index()
            
            # Save once at the end
            self._save_faiss_index()
            
            logging.info(f"✅ FAISS index rebuilt successfully ({total_processed} chunks).")

    # --------------------------
    # Document Management
    # --------------------------

    def compute_file_hash(self, file_path: str) -> str:
        """Compute SHA256 hash of file for deduplication."""
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            while chunk := f.read(8192):
                sha256.update(chunk)
        return sha256.hexdigest()

    def document_exists(self, file_hash: str) -> bool:
        """Check if document already exists in database."""
        with self._connect() as con:
            row = con.execute(
                "SELECT 1 FROM documents WHERE file_hash = ? LIMIT 1",
                (file_hash,)
            ).fetchone()
        return row is not None

    def add_document(
        self,
        file_hash: str,
        filename: str,
        file_type: str,
        file_size: int,
        page_count: Optional[int],
        chunks: List[Dict],  # [{'text': str, 'embedding': np.ndarray, 'page_number': int}, ...]
        upload_source: str = "web_ui"
    ) -> int:
        """
        Add document and its chunks to database with FAISS indexing.
        """
        timestamp = int(time.time())
        chunk_embeddings = []
        chunk_ids = []

        with self._connect() as con:
            # Insert document metadata
            cur = con.execute("""
                INSERT INTO documents (
                    file_hash, filename, file_type, file_size, 
                    page_count, chunk_count, upload_source, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                file_hash,
                filename,
                file_type,
                file_size,
                page_count,
                len(chunks),
                upload_source,
                timestamp
            ))
            document_id = cur.lastrowid

            # Insert chunks and collect embeddings
            for idx, chunk in enumerate(chunks):
                # Ensure embedding is a list for JSON serialization
                emb_list = chunk['embedding'].tolist() if isinstance(chunk['embedding'], np.ndarray) else chunk['embedding']
                
                con.execute("""
                    INSERT INTO chunks (
                        document_id, chunk_index, text,
                        page_number, created_at, embedding
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    document_id,
                    idx,
                    chunk['text'],
                    chunk.get('page_number'),
                    timestamp,
                    json.dumps(emb_list)
                ))
                
                # Store embedding for FAISS (as numpy array)
                chunk_embeddings.append(np.array(emb_list, dtype='float32'))
                # Get the chunk ID (we'll need to query it after commit)
                chunk_ids.append(None)  # Will be filled after commit

            con.commit()

            # Get actual chunk IDs
            chunk_rows = con.execute("""
                SELECT id FROM chunks 
                WHERE document_id = ? 
                ORDER BY chunk_index
            """, (document_id,)).fetchall()
            
            chunk_ids = [row[0] for row in chunk_rows]

        # Add embeddings to FAISS
        self._add_embeddings_to_faiss(chunk_embeddings, chunk_ids, save_index=True)
        
        return document_id

    def list_documents(self, limit: int = 1000) -> List[Dict]:
        """List all documents."""
        with self._connect() as con:
            rows = con.execute("""
                SELECT id, filename, file_type, page_count, chunk_count, created_at
                FROM documents
                ORDER BY created_at DESC
                LIMIT ?
            """, (limit,)).fetchall()
        
        return [
            {
                "id": r[0],
                "filename": r[1],
                "file_type": r[2],
                "page_count": r[3],
                "chunk_count": r[4],
                "created_at": r[5]
            }
            for r in rows
        ]

    def get_document(self, document_id: int) -> Optional[Dict]:
        """Retrieve a single document's metadata by ID."""
        with self._connect() as con:
            row = con.execute("""
                SELECT id, filename, file_type, page_count, chunk_count, created_at
                FROM documents WHERE id = ?
            """, (document_id,)).fetchone()
            if row:
                return {"id": row[0], "filename": row[1], "file_type": row[2], "page_count": row[3], "chunk_count": row[4], "created_at": row[5]}
        return None

    def delete_document(self, document_id: int) -> bool:
        """Delete document and all its chunks (and remove from FAISS)."""
        with self._connect() as con:
            # Check if document exists
            exists = con.execute(
                "SELECT 1 FROM documents WHERE id = ?",
                (document_id,)
            ).fetchone()

            if not exists:
                return False

            # Get chunk IDs to remove from FAISS
            chunk_rows = con.execute("""
                SELECT id FROM chunks WHERE document_id = ?
            """, (document_id,)).fetchall()
            
            chunk_ids_to_remove = [row[0] for row in chunk_rows]

            # Delete from SQLite
            con.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
            con.execute("DELETE FROM documents WHERE id = ?", (document_id,))
            con.commit()

        # Remove from FAISS immediately
        if self.faiss_index and chunk_ids_to_remove:
            with self.index_lock:
                self.faiss_index.remove_ids(np.array(chunk_ids_to_remove).astype('int64'))
                self._save_faiss_index()

        return True

    def find_broken_documents(self) -> List[Dict]:
        """Find documents with integrity issues (0 chunks, count mismatch, missing embeddings)."""
        broken = []
        with self._connect() as con:
            # Check 1: Metadata mismatch or empty chunks
            rows = con.execute("""
                SELECT d.id, d.filename, d.chunk_count, COUNT(c.id) as actual_chunks
                FROM documents d
                LEFT JOIN chunks c ON d.id = c.document_id
                GROUP BY d.id
                HAVING d.chunk_count != actual_chunks OR d.chunk_count = 0
            """).fetchall()
            
            for r in rows:
                issue = "No chunks found" if r[2] == 0 else f"Chunk count mismatch (Meta: {r[2]}, Actual: {r[3]})"
                broken.append({'id': r[0], 'filename': r[1], 'issue': issue})

            # Check 2: Missing embeddings
            rows_emb = con.execute("""
                SELECT d.id, d.filename, COUNT(c.id)
                FROM documents d
                JOIN chunks c ON d.id = c.document_id
                WHERE c.embedding IS NULL
                GROUP BY d.id
            """).fetchall()

            for r in rows_emb:
                if not any(b['id'] == r[0] for b in broken):
                    broken.append({'id': r[0], 'filename': r[1], 'issue': f"Missing embeddings for {r[2]} chunks"})
        
        return broken

    def get_orphaned_chunk_count(self) -> int:
        """Count chunks that have no parent document."""
        with self._connect() as con:
            row = con.execute("""
                SELECT COUNT(c.id) 
                FROM chunks c 
                LEFT JOIN documents d ON c.document_id = d.id 
                WHERE d.id IS NULL
            """).fetchone()
        return row[0] if row else 0

    def delete_orphaned_chunks(self) -> int:
        """Delete chunks that have no parent document."""
        chunk_ids_to_remove = []
        with self._connect() as con:
            # Get IDs first
            rows = con.execute("""
                SELECT id FROM chunks 
                WHERE document_id NOT IN (SELECT id FROM documents)
            """).fetchall()
            chunk_ids_to_remove = [r[0] for r in rows]
            
            cur = con.execute("""
                DELETE FROM chunks 
                WHERE document_id NOT IN (SELECT id FROM documents)
            """)
            con.commit()
            count = cur.rowcount
        
        if count > 0 and self.faiss_index and chunk_ids_to_remove:
            with self.index_lock:
                self.faiss_index.remove_ids(np.array(chunk_ids_to_remove).astype('int64'))
                self._save_faiss_index()
            
        return count

    def optimize(self):
        """Rebuild FAISS index to remove ghost vectors (deleted documents)."""
        self._rebuild_index()
        self.vacuum()

    def vacuum(self):
        """Reclaim unused disk space in SQLite."""
        try:
            with self._connect() as con:
                con.execute("VACUUM")
        except Exception as e:
            logging.warning(f"⚠️ Document vacuum failed: {e}")

    def clear(self):
        """DANGEROUS: Clear all documents and chunks."""
        with self._connect() as con:
            con.execute("DELETE FROM chunks")
            con.execute("DELETE FROM documents")
            con.commit()
        
        # Clear FAISS index
        dimension = self._detect_embedding_dimension()
        self.faiss_index = faiss.IndexIDMap(faiss.IndexFlatIP(dimension))
        self._save_faiss_index()

    def get_total_documents(self) -> int:
        with self._connect() as con:
            return con.execute("SELECT COUNT(*) FROM documents").fetchone()[0]

    def get_total_chunks(self) -> int:
        with self._connect() as con:
            return con.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    def search(self, query_embedding: List[float], limit: int = 5) -> List[Dict]:
        """
        Search for similar chunks using the query embedding.
        """
        if not self.faiss_index or self.faiss_index.ntotal == 0:
            return []

        # Prepare query vector
        query_vector = np.array(query_embedding, dtype='float32').reshape(1, -1)
        faiss.normalize_L2(query_vector)

        # Search FAISS
        scores, indices = self.faiss_index.search(query_vector, limit)

        # Filter valid indices and map scores
        valid_indices = [int(idx) for idx in indices[0] if idx != -1]
        if not valid_indices:
            return []
            
        id_to_score = {int(idx): float(score) for idx, score in zip(indices[0], scores[0]) if idx != -1}
        placeholders = ','.join('?' for _ in valid_indices)

        results = []
        with self._connect() as con:
            rows = con.execute(f"""
                SELECT c.id, c.text, d.filename, c.page_number 
                FROM chunks c 
                JOIN documents d ON c.document_id = d.id 
                WHERE c.id IN ({placeholders})
            """, valid_indices).fetchall()
            
            for r in rows:
                chunk_id = r[0]
                results.append({
                    "id": chunk_id,
                    "content": r[1],
                    "source": r[2],
                    "page": r[3],
                    "score": id_to_score.get(chunk_id, 0.0)
                })
        
        # Sort by score descending (SQL IN does not guarantee order)
        results.sort(key=lambda x: x["score"], reverse=True)
        return results

    def search_keyword(self, query: str, limit: int = 5) -> List[Dict]:
        """
        Search for chunks using SQLite FTS5 (Keyword Search).
        """
        if not query.strip():
            return []
        
        # Basic sanitization for FTS5
        # Remove characters that might interfere with standard query syntax if not intended
        safe_query = query.replace('"', '').replace("'", "").replace("*", "")
        
        results = []
        with self._connect() as con:
            try:
                # chunks_fts is virtual table. rowid maps to chunks.id
                # We use the 'rank' column for ordering (lower is better in FTS5)
                rows = con.execute("""
                    SELECT c.id, c.text, d.filename, c.page_number, chunks_fts.rank
                    FROM chunks_fts
                    JOIN chunks c ON c.id = chunks_fts.rowid
                    JOIN documents d ON c.document_id = d.id
                    WHERE chunks_fts MATCH ?
                    ORDER BY chunks_fts.rank
                    LIMIT ?
                """, (safe_query, limit)).fetchall()
                
                for r in rows:
                    results.append({
                        "id": r[0],
                        "content": r[1],
                        "source": r[2],
                        "page": r[3],
                        "score": r[4] # FTS rank
                    })
            except sqlite3.OperationalError as e:
                logging.warning(f"FTS5 Search failed: {e}")
                return []
        return results

    def search_hybrid(self, query_text: str, query_embedding: List[float], limit: int = 5, rrf_k: int = 60) -> List[Dict]:
        """
        Combine Vector Search and Keyword Search using Reciprocal Rank Fusion (RRF).
        """
        # 1. Get Vector Results
        vector_results = self.search(query_embedding, limit=limit)
        
        # 2. Get Keyword Results
        keyword_results = self.search_keyword(query_text, limit=limit)
        
        # 3. RRF Fusion
        scores = {}
        chunk_map = {}
        
        def add_scores(results):
            for rank, res in enumerate(results):
                cid = res['id']
                if cid not in scores:
                    scores[cid] = 0.0
                    chunk_map[cid] = res
                scores[cid] += 1.0 / (rrf_k + rank + 1)

        add_scores(vector_results)
        add_scores(keyword_results)
        
        # Sort by RRF score
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        
        final_results = []
        for cid in sorted_ids[:limit]:
            item = chunk_map[cid]
            item['rrf_score'] = scores[cid]
            final_results.append(item)
            
        return final_results

# Global instance
document_store = FaissDocumentStore()