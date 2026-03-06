import numpy as np
import json
import logging
from typing import List, Dict, Tuple, Set
from core.llm import LLMBridge
from core.settings import settings
from .backend import memory_store
import asyncio
from functools import partial

logger = logging.getLogger(__name__)

DEFAULT_ENTAILMENT_TIMEOUT = 10


class MemoryConsolidator:
    """
    Identifies semantic duplicates and consolidates them using parent_id chaining.
    Uses FAISS for efficient similarity search when available.
    """

    def __init__(self, config: Dict = None):
        self.store = memory_store
        self.config = config or {}
        self.llm = LLMBridge(
            base_url=settings.get("llm_api_url"),
            api_key=settings.get("llm_api_key")
        )

    def _fetch_candidates_with_timestamps(self) -> List[Dict]:
        """Fetch candidate memories with timestamps for consolidation."""
        with self.store._connect() as con:
            rows = con.execute("""
                SELECT id, text, created_at 
                FROM memories 
                WHERE deleted = 0 AND parent_id IS NULL AND embedding IS NOT NULL
                ORDER BY created_at DESC
            """).fetchall()

        return [{"id": r[0], "text": r[1], "created_at": r[2]} for r in rows]

    def _find_similar_pairs_faiss(self, threshold: float) -> List[Tuple[int, int, float]]:
        """Use FAISS to find similar memory pairs above threshold."""
        if not self.store.faiss_index or self.store.faiss_index.ntotal == 0:
            return []

        with self.store._connect() as con:
            rows = con.execute("""
                SELECT id, embedding FROM memories 
                WHERE deleted = 0 AND parent_id IS NULL AND embedding IS NOT NULL
                ORDER BY id ASC
            """).fetchall()

        embeddings = []
        id_to_idx = {}
        for idx, r in enumerate(rows):
            emb = self.store._parse_embedding(r[1])
            if emb is not None:
                embeddings.append(emb)
                id_to_idx[r[0]] = idx

        if len(embeddings) < 2:
            return []

        matrix = np.stack(embeddings).astype(np.float64)
        norm = np.linalg.norm(matrix, axis=1, keepdims=True)
        matrix_norm = matrix / (norm + 1e-10)
        sim_matrix = np.dot(matrix_norm, matrix_norm.T)

        pairs = []
        ids = list(id_to_idx.keys())
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                if sim_matrix[i, j] >= threshold:
                    pairs.append((ids[i], ids[j], float(sim_matrix[i, j])))

        return pairs

    def _get_processed_pairs(self) -> Set[Tuple[int, int]]:
        """Get already-processed consolidation pairs from meta_memories for idempotency."""
        processed = set()
        try:
            with self.store._connect() as con:
                rows = con.execute("""
                    SELECT target_ids FROM meta_memories 
                    WHERE action = 'consolidate'
                """).fetchall()

            for row in rows:
                try:
                    target_ids = json.loads(row[0])
                    if len(target_ids) == 2:
                        pair = tuple(sorted([target_ids[0], target_ids[1]]))
                        processed.add(pair)
                except (json.JSONDecodeError, IndexError):
                    continue
        except Exception as e:
            logger.warning(f"Failed to load processed consolidation pairs: {e}")

        return processed

    def _log_consolidation(self, child_id: int, parent_id: int, text_a: str, text_b: str):
        """Log consolidation to meta_memories for idempotency tracking."""
        self.store.log_meta_memory(
            action="consolidate",
            target_ids=[child_id, parent_id],
            new_value=json.dumps({"text_a": text_a[:100], "text_b": text_b[:100]}),
            description=f"Consolidated memory #{child_id} into #{parent_id}"
        )

    async def run(self) -> int:
        """Run consolidation process. Returns number of memories consolidated."""
        THRESHOLD = float(self.config.get("consolidation_threshold", 0.92))

        processed_pairs = self._get_processed_pairs()
        logger.info(f"Starting consolidation (threshold: {THRESHOLD}, already processed: {len(processed_pairs)} pairs)")

        loop = asyncio.get_running_loop()
        memories = await loop.run_in_executor(self.store.executor, self._fetch_candidates_with_timestamps)

        if not memories:
            logger.info("No memories to consolidate")
            return 0

        id_to_memory = {m["id"]: m for m in memories}

        similar_pairs = await loop.run_in_executor(
            self.store.executor,
            partial(self._find_similar_pairs_faiss, threshold=THRESHOLD)
        )

        logger.info(f"Found {len(similar_pairs)} similar pairs to check")

        consolidated_count = 0

        for mem_a_id, mem_b_id, similarity in similar_pairs:
            pair = tuple(sorted([mem_a_id, mem_b_id]))
            if pair in processed_pairs:
                continue

            mem_a = id_to_memory.get(mem_a_id)
            mem_b = id_to_memory.get(mem_b_id)

            if not mem_a or not mem_b:
                continue

            # Determine older vs newer based on created_at timestamp (not ID)
            if mem_a["created_at"] >= mem_b["created_at"]:
                older_mem = mem_b
                newer_mem = mem_a
            else:
                older_mem = mem_a
                newer_mem = mem_b

            # Check entailment with timeout
            is_equivalent = await self._check_entailment_with_timeout(
                older_mem["text"], 
                newer_mem["text"]
            )

            if is_equivalent:
                await loop.run_in_executor(
                    self.store.executor, 
                    partial(self.store.set_parent, child_id=older_mem["id"], parent_id=newer_mem["id"])
                )

                self._log_consolidation(
                    older_mem["id"], 
                    newer_mem["id"],
                    older_mem["text"],
                    newer_mem["text"]
                )

                processed_pairs.add(pair)
                consolidated_count += 1
                logger.info(f"Consolidated: '{older_mem['text'][:30]}...' -> '{newer_mem['text'][:30]}...'")

        logger.info(f"Consolidation complete: {consolidated_count} memories consolidated")
        return consolidated_count

    async def _check_entailment_with_timeout(self, text_a: str, text_b: str, timeout: float = DEFAULT_ENTAILMENT_TIMEOUT) -> bool:
        """LLM entailment check with timeout protection."""
        try:
            return await asyncio.wait_for(
                self._check_entailment(text_a, text_b),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            logger.warning(f"LLM entailment check timed out after {timeout}s")
            return False
        except Exception as e:
            logger.warning(f"LLM entailment check failed: {e}")
            return False

    async def _check_entailment(self, text_a: str, text_b: str) -> bool:
        """Uses LLM to verify if two texts are semantically equivalent."""
        prompt = (
            f"Analyze these two memory entries:\n"
            f"A: {text_a}\n"
            f"B: {text_b}\n\n"
            "Are they semantically equivalent (do they represent the same fact/concept)?\n"
            "Ignore minor wording differences.\n"
            "Output ONLY 'YES' or 'NO'."
        )

        response = await self.llm.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=5
        )

        try:
            content = response["choices"][0]["message"]["content"].strip().upper()
            return "YES" in content
        except (KeyError, IndexError, AttributeError) as e:
            logger.warning(f"Failed to parse LLM entailment response: {e}")
            return False

