import numpy as np
import json
from typing import List, Dict, Tuple
from core.llm import LLMBridge
from core.settings import settings
from .backend import memory_store

class MemoryConsolidator:
    """
    Identifies semantic duplicates and consolidates them using parent_id chaining.
    """
    
    def __init__(self, config: Dict = None):
        self.store = memory_store
        self.config = config or {}
        self.llm = LLMBridge(
            base_url=settings.get("llm_api_url"),
            api_key=settings.get("llm_api_key")
        )

    async def run(self) -> int:
        """
        Runs the consolidation process.
        Returns the number of memories consolidated.
        """
        # 1. Fetch all active, un-consolidated memories
        with self.store._connect() as con:
            rows = con.execute("""
                SELECT id, text, embedding 
                FROM memories 
                WHERE deleted = 0 AND parent_id IS NULL AND embedding IS NOT NULL
                ORDER BY id ASC
            """).fetchall()
            
        if len(rows) < 2:
            return 0
            
        memories = []
        embeddings = []
        
        for r in rows:
            emb = self.store._parse_embedding(r[2])
            if emb is not None:
                memories.append({"id": r[0], "text": r[1]})
                embeddings.append(emb)
                
        # 2. Compute Similarity Matrix
        # Stack embeddings (N x D)
        matrix = np.stack(embeddings).astype(np.float64)
        
        # Normalize (L2)
        norm = np.linalg.norm(matrix, axis=1, keepdims=True)
        matrix_norm = matrix / (norm + 1e-10)
        
        # Dot product (N x N)
        sim_matrix = np.dot(matrix_norm, matrix_norm.T)
        
        consolidated_count = 0
        processed_ids = set()
        
        # 3. Iterate and Check
        # Threshold: 0.92 (Very strict semantic similarity)
        THRESHOLD = float(self.config.get("consolidation_threshold", 0.92))
        
        for i in range(len(memories)):
            if memories[i]["id"] in processed_ids: continue
            
            for j in range(i + 1, len(memories)):
                if memories[j]["id"] in processed_ids: continue
                
                similarity = sim_matrix[i, j]
                
                if similarity >= THRESHOLD:
                    # High similarity found. Verify with LLM.
                    mem_a = memories[i]
                    mem_b = memories[j]
                    
                    is_equivalent = await self._check_entailment(mem_a["text"], mem_b["text"])
                    
                    if is_equivalent:
                        # Consolidate!
                        # Strategy: Keep the longer/more detailed one, or just the newer one?
                        # Simple strategy: Keep A (older) as parent? No, usually we want the newest info.
                        # Let's keep the one that is longer (more info), or default to A.
                        
                        # Actually, Binah suggests version chaining. 
                        # Let's assume B is newer (higher ID usually).
                        # We set A's parent to B. A becomes "history". B is the "current".
                        
                        self.store.set_parent(child_id=mem_a["id"], parent_id=mem_b["id"])
                        processed_ids.add(mem_a["id"])
                        consolidated_count += 1
                        print(f"🔗 Consolidated: '{mem_a['text'][:30]}...' -> '{mem_b['text'][:30]}...'")
                        break # Move to next i
                        
        return consolidated_count

    async def _check_entailment(self, text_a: str, text_b: str) -> bool:
        """
        Uses LLM to verify if two texts are semantically equivalent.
        """
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
        except:
            return False