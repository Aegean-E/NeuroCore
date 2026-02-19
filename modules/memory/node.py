import asyncio
import json
import os
import time
from core.llm import LLMBridge
from core.settings import settings
from .backend import memory_store
from .arbiter import MemoryArbiter
from .consolidation import MemoryConsolidator

class ConfigLoader:
    _cache = {"mtime": 0, "data": {}}
    _path = os.path.join(os.path.dirname(__file__), "module.json")

    @classmethod
    def get_config(cls):
        try:
            if os.path.exists(cls._path):
                mtime = os.path.getmtime(cls._path)
                if mtime > cls._cache["mtime"]:
                    with open(cls._path, "r") as f:
                        cls._cache["data"] = json.load(f).get("config", {})
                    cls._cache["mtime"] = mtime
        except Exception as e:
            print(f"Error loading memory config: {e}")
        return cls._cache["data"]

class MemoryRecallExecutor:
    def __init__(self):
        self.config = ConfigLoader.get_config()

    async def receive(self, input_data: dict, config: dict = None) -> dict:
        config = config or {}
        messages = input_data.get("messages", [])
        if not messages:
            return input_data

        # Get the last user message to use as a query
        last_user_msg = next((m for m in reversed(messages) if m["role"] == "user"), None)
        if not last_user_msg:
            return input_data

        query_text = last_user_msg["content"]
        
        # Generate embedding
        llm_bridge = LLMBridge(
            base_url=settings.get("llm_api_url"), 
            api_key=settings.get("llm_api_key"),
            embedding_base_url=settings.get("embedding_api_url"),
            embedding_model=settings.get("embedding_model")
        )
        embedding = await llm_bridge.get_embedding(query_text)
        
        if not embedding:
            return input_data

        # Search memory
        limit_val = config.get("limit")
        if limit_val is None:
            limit_val = self.config.get("recall_limit", 3)
        try:
            limit = int(limit_val)
            if limit < 1:
                limit = 1
        except (ValueError, TypeError):
            limit = 3

        min_score_val = config.get("min_score")
        if min_score_val is None:
            min_score_val = self.config.get("recall_min_score", 0.0)
        try:
            min_score = float(min_score_val)
            if min_score < 0.0: min_score = 0.0
            if min_score > 1.0: min_score = 1.0
        except (ValueError, TypeError):
            min_score = 0.0

        results = memory_store.search(embedding, limit=limit)
        
        # Filter results by minimum score if configured
        results = [r for r in results if r.get('score', 0) >= min_score]
        
        if not results:
            return input_data

        # Format context
        context_str = "\n".join([f"- {r['text']}" for r in results])
        system_msg = {
            "role": "system", 
            "content": f"Relevant memories retrieved from database:\n{context_str}\nUse these to answer the user if relevant."
        }

        # Inject into messages (before the last message)
        new_messages = messages[:-1] + [system_msg] + [messages[-1]]
        
        return {**input_data, "messages": new_messages}

    async def send(self, processed_data: dict) -> dict:
        return processed_data

class MemorySaveExecutor:
    def __init__(self):
        # Load config from module.json to ensure we have the latest settings
        self.config = ConfigLoader.get_config()
        self.arbiter = MemoryArbiter(memory_store, config=self.config)

    async def _save_background(self, text: str, subject: str = "User", confidence: float = 1.0):
        """Handles embedding generation and saving in the background."""
        try:
            llm_bridge = LLMBridge(
                base_url=settings.get("llm_api_url"), 
                api_key=settings.get("llm_api_key"),
                embedding_base_url=settings.get("embedding_api_url"),
                embedding_model=settings.get("embedding_model")
            )
            embedding = await llm_bridge.get_embedding(text)
            if embedding:
                self.arbiter.consider(
                    text=text,
                    confidence=confidence,
                    subject=subject,
                    embedding=embedding
                )
        except Exception as e:
            print(f"Background memory save failed: {e}")

    async def receive(self, input_data: dict, config: dict = None) -> dict:
        # Pass through data immediately, save in background (conceptually)
        # For now, we await to ensure it saves.
        config = config or {}
        
        text_to_save = None
        subject = "User"

        # 1. Check for direct content (e.g., from Chat Output) -> Assistant Memory
        if "content" in input_data and "messages" not in input_data:
            text_to_save = input_data["content"]
            subject = "Assistant"
        
        # 2. Check for OpenAI-style response (from LLM Core) -> Assistant Memory
        elif "choices" in input_data and isinstance(input_data["choices"], list):
            try:
                text_to_save = input_data["choices"][0]["message"]["content"]
                subject = "Assistant"
            except (IndexError, KeyError, TypeError):
                pass

        # 3. Check for message history (e.g., from Chat Input) -> User Memory
        elif "messages" in input_data:
            messages = input_data.get("messages", [])
            last_user_msg = next((m for m in reversed(messages) if m["role"] == "user"), None)
            if last_user_msg:
                text_to_save = last_user_msg["content"]
                subject = "User"

        # Determine default confidence
        # We default Assistant memories to 0.5 so they are filtered out by the default 0.85 threshold.
        # User memories remain at 1.0 (or configured default).
        base_default = float(self.config.get("save_default_confidence", 1.0))
        default_conf = min(base_default, 0.5) if subject == "Assistant" else base_default

        try:
            confidence = float(config.get("confidence") if config.get("confidence") is not None else default_conf)
        except (ValueError, TypeError):
            confidence = default_conf

        if text_to_save and len(text_to_save.strip()) > 10:
            # Fire and forget: Don't block the flow for embedding generation
            asyncio.create_task(self._save_background(
                text_to_save, 
                subject=subject, 
                confidence=confidence
            ))
            
            # Auto-Consolidation Check
            try:
                auto_hours = float(self.config.get("auto_consolidation_hours", 24))
                if auto_hours > 0:
                    if time.time() - memory_store.last_consolidation_ts > (auto_hours * 3600):
                        print("⏳ Triggering Auto-Consolidation...")
                        memory_store.last_consolidation_ts = time.time()
                        asyncio.create_task(MemoryConsolidator(config=self.config).run())
            except Exception as e:
                print(f"Auto-consolidation check failed: {e}")

        return input_data

    async def send(self, processed_data: dict) -> dict:
        return processed_data

async def get_executor_class(node_type_id: str):
    if node_type_id == "memory_recall":
        return MemoryRecallExecutor
    if node_type_id == "memory_save":
        return MemorySaveExecutor
    return None