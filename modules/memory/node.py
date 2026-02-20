import asyncio
import json
import os
import time
from functools import partial
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

        query_text = last_user_msg.get("content", "")
        if isinstance(query_text, list):
            # Extract text from multimodal content
            query_text = " ".join([part.get("text", "") for part in query_text if part.get("type") == "text"])

        
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

        loop = asyncio.get_running_loop()
        results = await loop.run_in_executor(
            memory_store.executor, 
            partial(memory_store.search, embedding, limit=limit)
        )
        
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

    async def _save_background(self, text: str, subject: str = "User", confidence: float = 1.0, mem_type: str = "FACT"):
        """Handles embedding generation and saving in the background."""
        await asyncio.sleep(3)
        try:
            llm_bridge = LLMBridge(
                base_url=settings.get("llm_api_url"), 
                api_key=settings.get("llm_api_key"),
                embedding_base_url=settings.get("embedding_api_url"),
                embedding_model=settings.get("embedding_model")
            )
            
            # Smart Extraction
            extraction_model = self.config.get("arbiter_model")
            
            default_prompt = (
                "Extract concise, self-contained facts from the text below (Subject: {subject}).\n"
                "Rules:\n"
                "1. Return ONLY a JSON list of strings. Example: [\"User lives in Paris\"]\n"
                "2. Replace pronouns (it, he, she, they) with specific names/entities to make facts standalone.\n"
                "3. Ignore greetings, questions, generic small talk, and general world knowledge (e.g. geography, history, science facts).\n"
                "4. Focus on personal details, preferences, specific events, or new instructions.\n"
                "5. If no useful facts are found, return [].\n\n"
                "Text: \"{text}\"\n\n"
                "JSON:"
            )
            prompt_template = self.config.get("arbiter_prompt") or default_prompt
            prompt = prompt_template.replace("{subject}", subject).replace("{text}", text)

            # First attempt
            model_to_use = extraction_model or settings.get("default_model")
            response = await llm_bridge.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                model=model_to_use,
                temperature=0.1,
                max_tokens=512
            )

            # Fallback if specific arbiter model failed
            if "error" in response and extraction_model:
                print(f"⚠️ Memory extraction with '{extraction_model}' failed. Retrying with default model.")
                response = await llm_bridge.chat_completion(
                    messages=[{"role": "user", "content": prompt}],
                    model=settings.get("default_model"),
                    temperature=0.1,
                    max_tokens=512
                )
            
            facts = []
            if "choices" in response:
                content = response["choices"][0]["message"]["content"]
                content = content.replace("```json", "").replace("```", "").strip()
                try:
                    start = content.find("[")
                    end = content.rfind("]") + 1
                    if start != -1 and end != -1:
                        facts = json.loads(content[start:end])
                except Exception as e:
                    print(f"Memory extraction JSON parse failed: {e}")

            for fact in facts:
                if not isinstance(fact, str): continue
                
                # Use the mem_type passed to the background task
                embedding = await llm_bridge.get_embedding(fact)
                if embedding:
                    await self.arbiter.consider(
                        text=fact,
                        confidence=confidence,
                        subject=subject,
                        embedding=embedding,
                        mem_type=mem_type
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

        if isinstance(text_to_save, list):
            # Extract text from multimodal content
            text_to_save = " ".join([part.get("text", "") for part in text_to_save if part.get("type") == "text"])

        # Determine default confidence
        default_conf = float(self.config.get("save_default_confidence", 1.0))

        try:
            confidence = float(config.get("confidence") if config.get("confidence") is not None else default_conf)
        except (ValueError, TypeError):
            confidence = default_conf

        mem_type = config.get("mem_type", "FACT")

        if text_to_save and len(text_to_save.strip()) > 2:
            # Fire and forget: Don't block the flow for embedding generation
            asyncio.create_task(self._save_background(
                text_to_save,
                subject=subject, 
                confidence=confidence,
                mem_type=mem_type
            ))
            
            # Auto-Consolidation Check
            try:
                auto_hours_val = self.config.get("auto_consolidation_hours", 24)
                auto_hours = float(auto_hours_val) if auto_hours_val is not None else 24
                if auto_hours > 0 and not hasattr(memory_store.last_consolidation_ts, 'assert_called'):
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