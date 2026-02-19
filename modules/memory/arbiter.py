from typing import Optional, Callable, Any, Dict, List
import numpy as np
import logging
import time
from datetime import datetime

from .backend import MemoryStore

DEFAULT_TYPE_PRECEDENCE = {
    "PERMISSION": 0,
    "RULE": 1,
    "IDENTITY": 2,
    "FACT": 3,
    "PREFERENCE": 4,
    "BELIEF": 5,
}

DEFAULT_CONFIDENCE_MIN = {
    "PERMISSION": 0.85,
    "RULE": 0.9,
    "IDENTITY": 0.8,
    "PREFERENCE": 0.6,
    "FACT": 0.7,
    "BELIEF": 0.5,
}

class MemoryArbiter:
    """
    Autonomous bridge between Reasoning and MemoryStore.
    Enforces admission rules and conflict resolution.
    """

    def __init__(self, memory_store: MemoryStore, embed_fn: Optional[Callable] = None, config: Dict = None):
        self.memory_store = memory_store
        self.embed_fn = embed_fn
        self.config = config or {}
        self.type_precedence = self.config.get("arbiter_precedence", DEFAULT_TYPE_PRECEDENCE)
        self.confidence_min = self.config.get("arbiter_confidence_thresholds", DEFAULT_CONFIDENCE_MIN)

    def consider(
        self,
        text: str,
        mem_type: str = "FACT",
        confidence: float = 1.0,
        subject: str = "User",
        source: str = "reasoning",
        embedding: Optional[List[float]] = None
    ) -> Optional[int]:
        """
        Decide whether to promote reasoning into memory.
        Returns memory_id if stored, None otherwise.
        """
        mem_type = mem_type.upper()
        
        # 0. Validate Type
        if mem_type not in self.type_precedence:
            # Default to FACT if unknown
            mem_type = "FACT"

        # 1. Confidence Gate
        min_conf = self.confidence_min.get(mem_type, 0.7)
        if confidence < min_conf:
            print(f"❌ [Arbiter] Confidence gate failed: {confidence} < {min_conf}")
            return None

        # 2. Identity & Duplicates
        identity = self.memory_store.compute_identity(text)
        
        # Check for exact duplicates in DB (backend handles this via identity check usually, 
        # but we can do a quick check here if needed, or rely on backend's return -1)
        
        # 3. Conflict Detection (Simplified)
        # In a full implementation, we would check for conflicting memories here.
        # For now, we rely on the backend's identity hash to prevent exact semantic duplicates.

        # 4. Save
        memory_id = self.memory_store.add_entry(
            text=text,
            embedding=embedding,
            confidence=confidence,
            subject=subject,
            mem_type=mem_type
        )

        if memory_id == -1:
            print(f"❌ [Arbiter] Duplicate or rejected by backend.")
            return None

        print(f"✅ [Arbiter] Memory saved: {mem_type} (ID: {memory_id})")
        return memory_id

    def consider_batch(self, candidates: List[Dict]) -> List[int]:
        """
        Process a batch of memory candidates.
        """
        promoted_ids = []
        for c in candidates:
            mid = self.consider(
                text=c.get("text", ""),
                mem_type=c.get("type", "FACT"),
                confidence=c.get("confidence", 1.0),
                subject=c.get("subject", "User"),
                source=c.get("source", "reasoning"),
                embedding=c.get("embedding")
            )
            if mid:
                promoted_ids.append(mid)
        return promoted_ids