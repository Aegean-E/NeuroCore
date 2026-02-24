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
from datetime import datetime
import concurrent.futures

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
        self.last_consolidation_ts = 0 # Track last consolidation time
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="MemoryWorker")
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
        
        # Self-healing: Ensure tables exist if the file was recently deleted/created
        res = con.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='memories'").fetchone()
        if not res:
            self._execute_init(con)
            
        try:
            yield con
            con.commit()
        except Exception:
            con.rollback()
            raise
        finally:
            con.close()

    def _init_db(self) -> None:
        with self._connect() as _:
            pass # _connect now handles initialization

    def _execute_init(self, con) -> None:
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
                deleted INTEGER DEFAULT 0,
                parent_id INTEGER DEFAULT NULL,
                source TEXT DEFAULT 'chat',
                verified INTEGER DEFAULT 0,
                expires_at INTEGER DEFAULT NULL,
                access_count INTEGER DEFAULT 0
            )
        """)
        con.execute("CREATE INDEX IF NOT EXISTS idx_identity ON memories(identity);")
        con.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON memories(created_at);")
        con.execute("CREATE INDEX IF NOT EXISTS idx_source ON memories(source);")
        
        # Migration logic
        try: con.execute("ALTER TABLE memories ADD COLUMN subject TEXT DEFAULT 'User'")
        except sqlite3.OperationalError: pass
        try: con.execute("ALTER TABLE memories ADD COLUMN parent_id INTEGER DEFAULT NULL")
        except sqlite3.OperationalError: pass
        try: con.execute("ALTER TABLE memories ADD COLUMN source TEXT DEFAULT 'chat'")
        except sqlite3.OperationalError: pass
        try: con.execute("ALTER TABLE memories ADD COLUMN verified INTEGER DEFAULT 0")
        except sqlite3.OperationalError: pass
        try: con.execute("ALTER TABLE memories ADD COLUMN expires_at INTEGER DEFAULT NULL")
        except sqlite3.OperationalError: pass
        try: con.execute("ALTER TABLE memories ADD COLUMN access_count INTEGER DEFAULT 0")
        except sqlite3.OperationalError: pass

        con.execute("""
            CREATE TABLE IF NOT EXISTS meta_memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                action TEXT NOT NULL,
                target_ids TEXT NOT NULL,
                new_value TEXT,
                description TEXT,
                created_at INTEGER NOT NULL
            )
        """)
        con.execute("CREATE INDEX IF NOT EXISTS idx_meta_action ON meta_memories(action);")
        con.execute("CREATE INDEX IF NOT EXISTS idx_meta_created_at ON meta_memories(created_at);")

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
            db_count = con.execute("SELECT COUNT(*) FROM memories WHERE deleted = 0 AND parent_id IS NULL AND embedding IS NOT NULL").fetchone()[0]
        
        if self.faiss_index.ntotal != db_count:
            print(f"Memory Index out of sync (Index: {self.faiss_index.ntotal}, DB: {db_count}). Rebuilding...")
            self._build_faiss_index()

    def _build_faiss_index(self):
        if not FAISS_AVAILABLE: return
        with self.faiss_lock:
            try:
                with self._connect() as con:
                    rows = con.execute("SELECT id, embedding FROM memories WHERE deleted = 0 AND parent_id IS NULL AND embedding IS NOT NULL ORDER BY id ASC").fetchall()
                
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

    def add_entry(self, text: str, embedding: Optional[List[float]] = None, confidence: float = 1.0, subject: str = "User", created_at: int = None, mem_type: str = "BELIEF", source: str = "chat", verified: bool = False, expires_at: int = None) -> int:
        identity = self.compute_identity(text)
        timestamp = created_at if created_at is not None else int(time.time())
        
        valid_types = ["BELIEF", "FACT", "RULE", "EXPERIENCE", "PREFERENCE"]
        final_type = mem_type if mem_type in valid_types else "BELIEF"
        
        # Convert belief to fact if verified
        if verified and final_type == "BELIEF":
            final_type = "FACT"
        
        # Apply TTL for beliefs (default 30 days if not specified)
        if final_type == "BELIEF" and expires_at is None:
            expires_at = timestamp + (30 * 24 * 60 * 60)  # 30 days
        
        final_source = source if source else "chat"
        emb_np = np.array(embedding, dtype='float32') if embedding else None
        embedding_blob = emb_np.tobytes() if emb_np is not None else None
        verified_int = 1 if verified else 0

        with self.write_lock:
            with self._connect() as con:
                # Check for duplicates
                exists = con.execute("SELECT 1 FROM memories WHERE identity = ? AND deleted = 0 AND parent_id IS NULL", (identity,)).fetchone()
                if exists:
                    return -1

                cur = con.execute("""
                    INSERT INTO memories (identity, type, subject, text, confidence, created_at, embedding, source, verified, expires_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (identity, final_type, subject, text, confidence, timestamp, embedding_blob, final_source, verified_int, expires_at))
                row_id = cur.lastrowid

            if FAISS_AVAILABLE and emb_np is not None and self.faiss_index:
                with self.faiss_lock:
                    emb_np_2d = emb_np.reshape(1, -1)
                    faiss.normalize_L2(emb_np_2d)
                    self.faiss_index.add_with_ids(emb_np_2d, np.array([row_id]).astype('int64'))
                    self._save_faiss_index()
            
            return row_id

    def set_parent(self, child_id: int, parent_id: int):
        """Links a memory to a parent, effectively archiving the child as an older version."""
        with self.write_lock:
            with self._connect() as con:
                con.execute("UPDATE memories SET parent_id = ? WHERE id = ?", (parent_id, child_id))
                # We do NOT delete the child. It remains for audit trails.
                # The search method filters out items with parent_id IS NOT NULL.
            
            # Remove from FAISS so it doesn't pollute search results
            if FAISS_AVAILABLE and self.faiss_index:
                with self.faiss_lock:
                    self.faiss_index.remove_ids(np.array([child_id]).astype('int64'))
                    self._save_faiss_index()

    def search(self, query_embedding: List[float], limit: int = 5, source_filter: str = None) -> List[Dict]:
        if not query_embedding: return []
        
        q_emb_np = np.array(query_embedding, dtype='float32')
        candidate_ids = []
        candidate_scores = {}
        current_time = int(time.time())

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
                    rows = con.execute("""
                        SELECT id, embedding FROM memories 
                        WHERE deleted = 0 AND parent_id IS NULL AND embedding IS NOT NULL
                        AND (expires_at IS NULL OR expires_at > ?)
                    """, (current_time,)).fetchall()
                
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
            query = f"SELECT id, text, created_at, type FROM memories WHERE id IN ({placeholders}) AND deleted = 0 AND parent_id IS NULL"
            params = list(top_ids)
            
            if source_filter:
                query += " AND source = ?"
                params.append(source_filter)
            
            with self._connect() as con:
                rows = con.execute(query, params).fetchall()
            
            valid_ids = []
            for r in rows:
                mid = r[0]
                mem_expires_at = r[2]  # This is created_at, need to get expires_at
                
                # Get expires_at for this memory
                expires_at = con.execute("SELECT expires_at FROM memories WHERE id = ?", (mid,)).fetchone()
                if expires_at and expires_at[0] and expires_at[0] < current_time:
                    continue  # Skip expired memories
                
                if mid in candidate_scores:
                    results.append({
                        "id": mid,
                        "text": r[1],
                        "created_at": r[2],
                        "type": r[3],
                        "score": candidate_scores[mid]
                    })
                    valid_ids.append(mid)
            
            # Increment access_count for returned memories
            if valid_ids:
                placeholders = ','.join(['?'] * len(valid_ids))
                with self._connect() as con:
                    con.execute(f"UPDATE memories SET access_count = access_count + 1 WHERE id IN ({placeholders})", valid_ids)
            
            results.sort(key=lambda x: x['score'], reverse=True)
            return results
        
        return []

    def get_recent(self, limit: int = 20) -> List[Dict]:
        with self._connect() as con:
            rows = con.execute("SELECT id, text, created_at FROM memories WHERE deleted = 0 AND parent_id IS NULL ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        return [{"id": r[0], "text": r[1], "created_at": r[2]} for r in rows]

    def browse(self, limit: int = 50, offset: int = 0, search_text: str = None, filter_date: str = "ALL", mem_type: str = None, source: str = None) -> List[Dict]:
        """
        Retrieves memories with optional filtering for the Memory Browser UI.
        """
        query = "SELECT id, subject, text, confidence, created_at, parent_id, type, source FROM memories WHERE deleted = 0 AND parent_id IS NULL"
        params = []
        
        if source:
            query += " AND source = ?"
            params.append(source)
        
        if filter_date == "TODAY":
            start_ts = int(datetime.now().replace(hour=0, minute=0, second=0, microsecond=0).timestamp())
            query += " AND created_at >= ?"
            params.append(start_ts)
        elif filter_date == "WEEK":
            start_ts = int(time.time()) - (7 * 24 * 60 * 60)
            query += " AND created_at >= ?"
            params.append(start_ts)
        elif filter_date == "MONTH":
            start_ts = int(time.time()) - (30 * 24 * 60 * 60)
            query += " AND created_at >= ?"
            params.append(start_ts)
            
        if search_text:
            query += " AND text LIKE ?"
            params.append(f"%{search_text}%")
            
        if mem_type:
            query += " AND type = ?"
            params.append(mem_type)
            
        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        
        with self._connect() as con:
            rows = con.execute(query, params).fetchall()
            
        return [
            {
                "id": r[0],
                "subject": r[1],
                "text": r[2],
                "confidence": r[3],
                "created_at": r[4],
                "is_consolidated": r[5] is not None,
                "type": r[6],
                "source": r[7] if len(r) > 7 else "chat"
            }
            for r in rows
        ]

    def find_similar(self, embedding: List[float], threshold: float = 0.9, limit: int = 5, exclude_source: str = None) -> List[Dict]:
        """
        Find memories with cosine similarity above threshold.
        Returns list of dicts with id, text, score.
        """
        if not embedding:
            return []
        
        q_emb_np = np.array(embedding, dtype='float32')
        candidate_ids = []
        candidate_scores = {}
        
        use_faiss = FAISS_AVAILABLE and self.faiss_index and self.faiss_index.ntotal > 0
        
        if use_faiss:
            with self.faiss_lock:
                q_emb_2d = q_emb_np.reshape(1, -1)
                faiss.normalize_L2(q_emb_2d)
                # Get more candidates to filter by threshold
                search_limit = max(limit * 4, 20)
                scores, indices = self.faiss_index.search(q_emb_2d, search_limit)
                
                for i, idx in enumerate(indices[0]):
                    if idx != -1:
                        score = float(scores[0][i])
                        if score >= threshold:
                            candidate_ids.append(int(idx))
                            candidate_scores[int(idx)] = score
        
        if candidate_ids:
            placeholders = ','.join(['?'] * len(candidate_ids))
            query = f"SELECT id, text FROM memories WHERE id IN ({placeholders}) AND deleted = 0 AND parent_id IS NULL"
            params = list(candidate_ids)
            
            if exclude_source:
                query += " AND source != ?"
                params.append(exclude_source)
            
            with self._connect() as con:
                rows = con.execute(query, params).fetchall()
            
            results = []
            for r in rows:
                mid = r[0]
                if mid in candidate_scores:
                    results.append({
                        "id": mid,
                        "text": r[1],
                        "score": candidate_scores[mid]
                    })
            
            results.sort(key=lambda x: x['score'], reverse=True)
            return results[:limit]
        
        return []

    def delete_entry(self, memory_id: int):
        with self.write_lock:
            with self._connect() as con:
                con.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            
            if FAISS_AVAILABLE and self.faiss_index:
                with self.faiss_lock:
                    self.faiss_index.remove_ids(np.array([memory_id]).astype('int64'))
                    self._save_faiss_index()

    def wipe_all(self):
        with self.write_lock:
            with self._connect() as con:
                con.execute("DELETE FROM memories")
                try: con.execute("DELETE FROM sqlite_sequence WHERE name='memories'")
                except Exception: pass
            
            if FAISS_AVAILABLE and self.faiss_index:
                with self.faiss_lock:
                    self.faiss_index.reset()
                    self._save_faiss_index(force=True)

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
            
            # Deleted (Hard deleted items cannot be counted, returning 0 for UI consistency)
            stats['deleted'] = 0
            
            # Consolidated (Hidden)
            row = con.execute("SELECT COUNT(*) FROM memories WHERE deleted = 0 AND parent_id IS NOT NULL").fetchone()
            stats['consolidated'] = row[0] if row else 0
            
        stats['grand_total'] = stats['total'] + stats['archived']
        return stats

    def log_meta_memory(self, action: str, target_ids: List[int], new_value: str = None, description: str = None) -> int:
        """Logs an action to the meta_memories table."""
        with self.write_lock:
            with self._connect() as con:
                cur = con.execute("""
                    INSERT INTO meta_memories (action, target_ids, new_value, description, created_at)
                    VALUES (?, ?, ?, ?, ?)
                """, (action, json.dumps(target_ids), new_value, description, int(time.time())))
                return cur.lastrowid

    def update_entry(self, memory_id: int, text: str = None, mem_type: str = None, verified: bool = None) -> Optional[int]:
        """
        Updates a memory entry. Archives the old version via parent_id.
        Returns the new memory ID, or None if not found.
        """
        with self.write_lock:
            with self._connect() as con:
                current = con.execute("""
                    SELECT text, type, verified FROM memories WHERE id = ? AND deleted = 0 AND parent_id IS NULL
                """, (memory_id,)).fetchone()
                
                if not current:
                    return None
                
                old_text, old_type, old_verified = current[0], current[1], current[2]
                
                new_text = text if text is not None else old_text
                new_type = mem_type if mem_type is not None else old_type
                new_verified = verified if verified is not None else bool(old_verified)
                
                con.execute("UPDATE memories SET parent_id = ? WHERE id = ?", (memory_id, memory_id))
                
                cur = con.execute("""
                    INSERT INTO memories (identity, type, subject, text, confidence, created_at, embedding, source, verified, parent_id)
                    SELECT identity, ?, subject, ?, confidence, ?, embedding, source, ?, NULL
                    FROM memories WHERE id = ?
                """, (new_type, new_text, int(time.time()), 1 if new_verified else 0, memory_id))
                new_id = cur.lastrowid
                
                changes = []
                if text and text != old_text:
                    changes.append(f"changed text from '{old_text}' to '{text}'")
                if mem_type and mem_type != old_type:
                    changes.append(f"changed type from '{old_type}' to '{mem_type}'")
                if verified is not None and verified != bool(old_verified):
                    changes.append(f"changed verified from {bool(old_verified)} to {verified}")
                
                description = f"Updated memory #{memory_id}: {', '.join(changes)}" if changes else f"Updated memory #{memory_id}"
                self.log_meta_memory("edit", [memory_id], json.dumps({"new_id": new_id}), description)
                
                if FAISS_AVAILABLE and self.faiss_index:
                    with self.faiss_lock:
                        self.faiss_index.remove_ids(np.array([memory_id]).astype('int64'))
                        self._save_faiss_index()
                
                return new_id

    def delete_entry(self, memory_id: int, reason: str = None):
        """Hard deletes a memory entry. Logs to meta_memories."""
        old_text = None
        with self._connect() as con:
            row = con.execute("SELECT text FROM memories WHERE id = ?", (memory_id,)).fetchone()
            if row:
                old_text = row[0]
        
        with self.write_lock:
            with self._connect() as con:
                con.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            
            if FAISS_AVAILABLE and self.faiss_index:
                with self.faiss_lock:
                    self.faiss_index.remove_ids(np.array([memory_id]).astype('int64'))
                    self._save_faiss_index()
            
            description = f"Deleted memory #{memory_id}" + (f": {reason}" if reason else "")
            self.log_meta_memory("delete", [memory_id], reason, description)

    def merge_memories(self, memory_ids: List[int], new_text: str, new_type: str = "BELIEF", new_verified: bool = False) -> Optional[int]:
        """
        Merges multiple memories into one. Archives originals via parent_id.
        Returns the new merged memory ID, or None if failed.
        """
        if not memory_ids or len(memory_ids) < 2:
            return None
        
        with self.write_lock:
            with self._connect() as con:
                originals = con.execute(f"""
                    SELECT id, text, embedding, identity FROM memories 
                    WHERE id IN ({','.join(['?'] * len(memory_ids))}) AND deleted = 0 AND parent_id IS NULL
                """, memory_ids).fetchall()
                
                if len(originals) < 2:
                    return None
                
                for orig_id, _, _, _ in originals:
                    con.execute("UPDATE memories SET parent_id = ? WHERE id = ?", (orig_id, orig_id))
                
                avg_embedding = None
                embeddings = []
                for _, _, emb, _ in originals:
                    if emb:
                        parsed = self._parse_embedding(emb)
                        if parsed is not None:
                            embeddings.append(parsed)
                
                if embeddings:
                    avg_embedding = np.mean(embeddings, axis=0)
                
                emb_np = avg_embedding if avg_embedding is not None else None
                embedding_blob = None
                if emb_np is not None:
                    embedding_blob = emb_np.tobytes()
                
                cur = con.execute("""
                    INSERT INTO memories (identity, type, subject, text, confidence, created_at, embedding, source, verified)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    self.compute_identity(new_text),
                    new_type,
                    "User",
                    new_text,
                    1.0,
                    int(time.time()),
                    embedding_blob,
                    "merge",
                    1 if new_verified else 0
                ))
                new_id = cur.lastrowid
                
                if FAISS_AVAILABLE and emb_np is not None and self.faiss_index:
                    with self.faiss_lock:
                        emb_np_2d = emb_np.reshape(1, -1)
                        faiss.normalize_L2(emb_np_2d)
                        self.faiss_index.add_with_ids(emb_np_2d, np.array([new_id]).astype('int64'))
                        self._save_faiss_index()
                
                description = f"Merged memories {', '.join(['#' + str(mid) for mid in memory_ids])} into #{new_id}: {new_text[:50]}..."
                self.log_meta_memory("merge", memory_ids, json.dumps({"new_id": new_id, "new_text": new_text}), description)
                
                return new_id

    def find_conflicts(self, memory_ids: List[int] = None) -> List[Dict]:
        """
        Finds contradictory memories using LLM.
        If memory_ids provided, checks only those. Otherwise checks all active memories.
        Returns list of dicts with conflicting memory pairs and LLM reason.
        """
        from core.llm import LLMBridge
        from core.settings import settings
        
        with self._connect() as con:
            if memory_ids:
                placeholders = ','.join(['?'] * len(memory_ids))
                rows = con.execute(f"""
                    SELECT id, text, type FROM memories 
                    WHERE id IN ({placeholders}) AND deleted = 0 AND parent_id IS NULL
                """, memory_ids).fetchall()
            else:
                rows = con.execute("""
                    SELECT id, text, type FROM memories 
                    WHERE deleted = 0 AND parent_id IS NULL
                """).fetchall()
        
        memories = [{"id": r[0], "text": r[1], "type": r[2]} for r in rows]
        
        if len(memories) < 2:
            return []
        
        conflicts = []
        client = LLMBridge(
            base_url=settings.get("llm_api_url"),
            api_key=settings.get("llm_api_key"),
            timeout=30.0
        )
        
        for i in range(len(memories)):
            for j in range(i + 1, len(memories)):
                mem1 = memories[i]
                mem2 = memories[j]
                
                prompt = f"""Do these two memories contradict each other?
Memory A (id={mem1['id']}): {mem1['text']}
Memory B (id={mem2['id']}): {mem2['text']}

Answer in this exact format:
CONFLICT: YES or NO
REASON: Brief explanation (1-2 sentences)"""
                
                try:
                    import asyncio
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        response = loop.run_until_complete(client.chat_completion(
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0,
                            max_tokens=100
                        ))
                        result_text = response.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
                    finally:
                        loop.close()
                    
                    if result_text.startswith("CONFLICT: YES"):
                        reason_start = result_text.find("REASON:") + 7
                        reason = result_text[reason_start:].strip() if reason_start > 6 else "Contradictory content"
                        
                        conflicts.append({
                            "memory_1": mem1,
                            "memory_2": mem2,
                            "reason": reason
                        })
                        
                        self.log_meta_memory("conflict", [mem1['id'], mem2['id']], None, f"Conflict detected: #{mem1['id']} contradicts #{mem2['id']}: {reason}")
                        
                except Exception as e:
                    print(f"Conflict check failed for {mem1['id']} vs {mem2['id']}: {e}")
        
        return conflicts

    def get_meta_memories(self, limit: int = 50, offset: int = 0) -> List[Dict]:
        """Retrieves meta-memory audit logs."""
        with self._connect() as con:
            rows = con.execute("""
                SELECT id, action, target_ids, new_value, description, created_at 
                FROM meta_memories ORDER BY created_at DESC LIMIT ? OFFSET ?
            """, (limit, offset)).fetchall()
        
        return [
            {
                "id": r[0],
                "action": r[1],
                "target_ids": json.loads(r[2]),
                "new_value": r[3],
                "description": r[4],
                "created_at": r[5]
            }
            for r in rows
        ]

# Global Instance
memory_store = MemoryStore()