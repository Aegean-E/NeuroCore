import os
import sqlite3
import time
import json
import threading
import hashlib
import logging
from typing import List, Dict, Optional, Tuple, Any
from contextlib import contextmanager
import numpy as np

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("Warning: FAISS not installed. Vector search will fall back to linear scan (slow).")

class MemoryStore:
    """
    Simplified MemoryStore for NeuroCore.
    """

    def __init__(self, db_path: str = "data/memory.sqlite3"):
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self.db_path = db_path
        self.write_lock = threading.Lock()
        self.faiss_lock = threading.Lock()
        self.unsaved_changes = 0
        self.save_threshold = 20  # Save index to disk after 20 changes
        self._init_db()
        
        self.faiss_index = None
        if FAISS_AVAILABLE:
            if not self._load_faiss_index():
                self._build_faiss_index()
            else:
                self._sync_faiss_index()

    @contextmanager
    def _connect(self):
        con = sqlite3.connect(self.db_path, check_same_thread=False)
        con.execute("PRAGMA journal_mode=WAL;")
        try:
            yield con
            con.commit()
        except Exception:
            con.rollback()
            raise
        finally:
            con.close()

    def _init_db(self) -> None:
        with self._connect() as con:
            con.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    identity TEXT NOT NULL,
                    type TEXT NOT NULL,
                    subject TEXT DEFAULT 'User',
                    text TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    created_at INTEGER NOT NULL,
                    embedding TEXT,
                    deleted INTEGER DEFAULT 0
                )
            """)
            con.execute("CREATE INDEX IF NOT EXISTS idx_identity ON memories(identity);")
            con.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON memories(created_at);")
            
            # Migration: Add subject column if it doesn't exist
            try:
                con.execute("ALTER TABLE memories ADD COLUMN subject TEXT DEFAULT 'User'")
            except sqlite3.OperationalError:
                pass

    def _parse_embedding(self, blob) -> Optional[np.ndarray]:
        if blob is None: return None
        try:
            if isinstance(blob, bytes):
                return np.frombuffer(blob, dtype='float32')
            return np.array(json.loads(blob), dtype='float32')
        except Exception as e:
            return None

    def _save_faiss_index(self, force=False):
        if not FAISS_AVAILABLE or not self.faiss_index: return
        
        self.unsaved_changes += 1
        if not force and self.unsaved_changes < self.save_threshold:
            return

        try:
            index_path = self.db_path.replace(".sqlite3", ".faiss")
            faiss.write_index(self.faiss_index, index_path)
            self.unsaved_changes = 0
        except Exception as e:
            print(f"Failed to save FAISS index: {e}")

    def _load_faiss_index(self) -> bool:
        if not FAISS_AVAILABLE: return False
        index_path = self.db_path.replace(".sqlite3", ".faiss")
        if os.path.exists(index_path):
            try:
                self.faiss_index = faiss.read_index(index_path)
                return True
            except Exception:
                return False
        return False

    def _sync_faiss_index(self):
        """Ensures FAISS index count matches DB count on startup."""
        if not FAISS_AVAILABLE or not self.faiss_index: return
        with self._connect() as con:
            db_count = con.execute("SELECT COUNT(*) FROM memories WHERE deleted = 0 AND embedding IS NOT NULL").fetchone()[0]
        
        if self.faiss_index.ntotal != db_count:
            print(f"Memory Index out of sync (Index: {self.faiss_index.ntotal}, DB: {db_count}). Rebuilding...")
            self._build_faiss_index()

    def _build_faiss_index(self):
        if not FAISS_AVAILABLE: return
        with self.faiss_lock:
            try:
                with self._connect() as con:
                    rows = con.execute("SELECT id, embedding FROM memories WHERE deleted = 0 AND embedding IS NOT NULL").fetchall()
                
                embeddings = []
                ids = []
                
                for r in rows:
                    emb = self._parse_embedding(r[1])
                    if emb is not None:
                        embeddings.append(emb)
                        ids.append(r[0])
                
                if not embeddings: return

                dimension = len(embeddings[0])
                embs_np = np.array(embeddings).astype('float32')
                faiss.normalize_L2(embs_np)
                ids_np = np.array(ids).astype('int64')

                quantizer = faiss.IndexFlatIP(dimension)
                self.faiss_index = faiss.IndexIDMap(quantizer)
                self.faiss_index.add_with_ids(embs_np, ids_np)
                self._save_faiss_index(force=True)
            except Exception as e:
                print(f"Failed to build FAISS index: {e}")

    def compute_identity(self, text: str) -> str:
        text_lower = " ".join(text.lower().strip().split())
        return hashlib.sha256(text_lower.encode("utf-8")).hexdigest()

    def add_entry(self, text: str, embedding: Optional[List[float]] = None, confidence: float = 1.0, subject: str = "User", mem_type: str = "FACT", created_at: int = None) -> int:
        identity = self.compute_identity(text)
        timestamp = created_at if created_at is not None else int(time.time())
        
        emb_np = np.array(embedding, dtype='float32') if embedding else None
        embedding_blob = emb_np.tobytes() if emb_np is not None else None

        with self.write_lock:
            with self._connect() as con:
                # Check for duplicates
                exists = con.execute("SELECT 1 FROM memories WHERE identity = ? AND deleted = 0", (identity,)).fetchone()
                if exists:
                    return -1

                cur = con.execute("""
                    INSERT INTO memories (identity, type, subject, text, confidence, created_at, embedding)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (identity, mem_type, subject, text, confidence, timestamp, embedding_blob))
                row_id = cur.lastrowid

            if FAISS_AVAILABLE and emb_np is not None and self.faiss_index:
                with self.faiss_lock:
                    emb_np_2d = emb_np.reshape(1, -1)
                    faiss.normalize_L2(emb_np_2d)
                    self.faiss_index.add_with_ids(emb_np_2d, np.array([row_id]).astype('int64'))
                    self._save_faiss_index()
            
            return row_id

    def search(self, query_embedding: List[float], limit: int = 5) -> List[Dict]:
        if not query_embedding: return []
        
        q_emb_np = np.array(query_embedding, dtype='float32')
        candidate_ids = []
        candidate_scores = {}

        # Determine if we can use FAISS
        use_faiss = FAISS_AVAILABLE and self.faiss_index and self.faiss_index.ntotal > 0

        # 1. FAISS Search
        if use_faiss:
            with self.faiss_lock:
                q_emb_2d = q_emb_np.reshape(1, -1)
                faiss.normalize_L2(q_emb_2d)
                scores, indices = self.faiss_index.search(q_emb_2d, limit * 2)
                
                for i, idx in enumerate(indices[0]):
                    if idx != -1:
                        candidate_ids.append(int(idx))
                        candidate_scores[int(idx)] = float(scores[0][i])

        # 2. Linear Fallback (If FAISS is missing or empty)
        else:
            try:
                with self._connect() as con:
                    rows = con.execute("SELECT id, embedding FROM memories WHERE deleted = 0 AND embedding IS NOT NULL").fetchall()
                
                # Normalize query
                q_norm = np.linalg.norm(q_emb_np)
                if q_norm > 0:
                    q_emb_np = q_emb_np / q_norm

                for r in rows:
                    emb = self._parse_embedding(r[1])
                    if emb is not None:
                        # Normalize target
                        e_norm = np.linalg.norm(emb)
                        if e_norm > 0:
                            emb = emb / e_norm
                        
                        score = np.dot(q_emb_np, emb)
                        candidate_ids.append(r[0])
                        candidate_scores[r[0]] = float(score)
            except Exception as e:
                print(f"Linear search failed: {e}")

        # 3. Fetch Details
        results = []
        if candidate_ids:
            # Sort by score descending and take top K to minimize DB fetch
            candidate_ids.sort(key=lambda x: candidate_scores.get(x, 0), reverse=True)
            top_ids = candidate_ids[:limit]
            
            placeholders = ','.join(['?'] * len(top_ids))
            with self._connect() as con:
                rows = con.execute(f"SELECT id, text, created_at FROM memories WHERE id IN ({placeholders}) AND deleted = 0", top_ids).fetchall()
            
            for r in rows:
                mid = r[0]
                if mid in candidate_scores:
                    results.append({
                        "id": mid,
                        "text": r[1],
                        "created_at": r[2],
                        "score": candidate_scores[mid]
                    })
            
            results.sort(key=lambda x: x['score'], reverse=True)
            return results
        
        return []

    def get_recent(self, limit: int = 20) -> List[Dict]:
        with self._connect() as con:
            rows = con.execute("SELECT id, text, created_at FROM memories WHERE deleted = 0 ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        return [{"id": r[0], "text": r[1], "created_at": r[2]} for r in rows]

    def browse(self, limit: int = 50, offset: int = 0, search_text: str = None, mem_type: str = None) -> List[Dict]:
        """
        Retrieves memories with optional filtering for the Memory Browser UI.
        """
        query = "SELECT id, type, subject, text, confidence, created_at FROM memories WHERE deleted = 0"
        params = []
        
        if mem_type and mem_type != "ALL":
            query += " AND type = ?"
            params.append(mem_type)
            
        if search_text:
            query += " AND text LIKE ?"
            params.append(f"%{search_text}%")
            
        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        
        with self._connect() as con:
            rows = con.execute(query, params).fetchall()
            
        return [
            {
                "id": r[0],
                "type": r[1],
                "subject": r[2],
                "text": r[3],
                "confidence": r[4],
                "created_at": r[5]
            }
            for r in rows
        ]

    def delete_entry(self, memory_id: int):
        with self.write_lock:
            with self._connect() as con:
                con.execute("UPDATE memories SET deleted = 1 WHERE id = ?", (memory_id,))
            
            if FAISS_AVAILABLE and self.faiss_index:
                with self.faiss_lock:
                    self.faiss_index.remove_ids(np.array([memory_id]).astype('int64'))
                    self._save_faiss_index()

    def get_memory_stats(self) -> Dict[str, Any]:
        """Returns statistics about the stored memories."""
        stats = {}
        with self._connect() as con:
            # Total active memories
            row = con.execute("SELECT COUNT(*) FROM memories WHERE deleted = 0").fetchone()
            stats['total'] = row[0] if row else 0
            
            # User vs Assistant
            row = con.execute("SELECT COUNT(*) FROM memories WHERE deleted = 0 AND subject = 'User'").fetchone()
            stats['user'] = row[0] if row else 0
            
            row = con.execute("SELECT COUNT(*) FROM memories WHERE deleted = 0 AND subject = 'Assistant'").fetchone()
            stats['assistant'] = row[0] if row else 0

            # Archived
            row = con.execute("SELECT COUNT(*) FROM memories WHERE deleted = 1").fetchone()
            stats['archived'] = row[0] if row else 0
            
            # Breakdown by type
            rows = con.execute("SELECT type, COUNT(*) FROM memories WHERE deleted = 0 GROUP BY type").fetchall()
            stats['types'] = {r[0]: r[1] for r in rows}
            
        return stats

# Global Instance
memory_store = MemoryStore()