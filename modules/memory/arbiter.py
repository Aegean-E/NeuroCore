from typing import Optional, Callable, Any, Dict, List
import numpy as np
import logging
import time
from datetime import datetime
import asyncio
from functools import partial

from .backend import MemoryStore
from core.llm import LLMBridge
from core.settings import settings

class MemoryArbiter:
    """
    Autonomous bridge between Reasoning and MemoryStore.
    Enforces admission rules and conflict resolution.
    """

    def __init__(self, memory_store: MemoryStore, embed_fn: Optional[Callable] = None, config: Dict = None):
        self.memory_store = memory_store
        self.embed_fn = embed_fn
        self.config = config or {}

    async def consider(
        self,
        text: str,
        confidence: float = 1.0,
        subject: str = "User",
        source: str = "reasoning",
        embedding: Optional[List[float]] = None,
        mem_type: str = "FACT"
    ) -> Optional[int]:
        """
        Decide whether to promote reasoning into memory.
        Returns memory_id if stored, None otherwise.
        """
        
        # 1. Confidence Gate
        min_conf = float(self.config.get("save_confidence_threshold", 0.75))
        if confidence < min_conf:
            print(f"❌ [Arbiter] Confidence gate failed: {confidence} < {min_conf}")
            return None

        # Type Validation: Default to FACT if unknown
        if mem_type not in ["FACT", "RULE", "PREFERENCE", "MEMORY"]:
            mem_type = "FACT"

        # 2. Identity & Duplicates
        identity = self.memory_store.compute_identity(text)
        
        # Check for exact duplicates in DB (backend handles this via identity check usually, 
        # but we can do a quick check here if needed, or rely on backend's return -1)
        
        # 3. Conflict Detection (Simplified)
        # In a full implementation, we would check for conflicting memories here.
        # For now, we rely on the backend's identity hash to prevent exact semantic duplicates.

        # 4. Save
        loop = asyncio.get_running_loop()
        memory_id = await loop.run_in_executor(
            self.memory_store.executor,
            partial(
                self.memory_store.add_entry,
                text=text,
                embedding=embedding,
                confidence=confidence,
                subject=subject,
                mem_type=mem_type
            )
        )

        if memory_id == -1:
            print(f"❌ [Arbiter] Duplicate or rejected by backend.")
            return None

        print(f"✅ [Arbiter] Memory saved (ID: {memory_id})")
        return memory_id

    async def consider_batch(self, candidates: List[Dict]) -> List[int]:
        """
        Process a batch of memory candidates.
        """
        promoted_ids = []
        for c in candidates:
            mid = await self.consider(
                text=c.get("text", ""),
                confidence=c.get("confidence", 1.0),
                subject=c.get("subject", "User"),
                source=c.get("source", "reasoning"),
                embedding=c.get("embedding")
            )
            if mid:
                promoted_ids.append(mid)
        return promoted_ids