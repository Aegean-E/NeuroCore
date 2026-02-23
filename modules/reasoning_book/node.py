from .service import service

VAGUE_PATTERNS = [
    "the assistant",
    "i will",
    "i am proceeding",
    "to further enhance",
    "let me proceed",
    "to continue improving",
    "this successful",
    "the search for",
    "was unsuccessful",
    "did not yield",
    "cognitive system",
    "improving internal reasoning",
    "complex pattern recognition",
    "advanced machine learning",
    "anomaly detection",
    "deep learning",
    "neural networks",
    "artificial intelligence",
    "machine learning",
    "improve internal",
    "foundational cognitive",
    "unify two abstract",
    "increase abstraction",
    "switch abstraction level",
    "adversarial self-critique",
    "rebuild from first principles",
    "there is no completion",
    "no explicit task",
    "self-generate",
    "the system aims",
    "need to refine",
    "focus on the",
    "consider",
    "deterministic principles",
    "probability distributions",
    "particles can exist",
]

MIN_CONTENT_LENGTH = 30


class ReasoningSaveExecutor:
    async def receive(self, data: dict, config: dict = None) -> dict:
        config = config or {}
        
        # Skip logging if there's an error in the data
        if isinstance(data, dict) and data.get("error"):
            return data
        
        content = None
        
        if isinstance(data, dict):
            # Only save from specific meaningful fields
            if "reasoning" in data:
                content = data["reasoning"]
            elif "thought" in data:
                content = data["thought"]
            elif "summary" in data:
                content = data["summary"]
            elif "conclusion" in data:
                content = data["conclusion"]
            elif "result" in data:
                content = data["result"]
        
        # If no meaningful content found, skip
        if not content:
            return data
        
        # Skip if too short
        if len(content) < MIN_CONTENT_LENGTH:
            return data
        
        # Skip vague/generic content
        content_lower = content.lower()
        for pattern in VAGUE_PATTERNS:
            if pattern in content_lower:
                return data
        
        service.log_thought(content, source=config.get("source", "Flow Node"))
        return data

    async def send(self, data: dict) -> dict:
        return data

class ReasoningLoadExecutor:
    async def receive(self, data: dict, config: dict = None) -> dict:
        config = config or {}
        last_n = int(config.get("last_n", 5))
        
        # Get thoughts from service (newest first)
        thoughts = service.get_thoughts()
        recent_thoughts = thoughts[:last_n]
        
        if isinstance(data, dict):
            result = data.copy()
            # Inject as a list of content strings
            result["reasoning_history"] = [t["content"] for t in recent_thoughts]
            # Inject as a formatted string for prompts (Chronological order)
            result["reasoning_context"] = "\n".join([f"[{t['timestamp']}] {t['content']}" for t in reversed(recent_thoughts)])
            return result
        return data

    async def send(self, data: dict) -> dict:
        return data

async def get_executor_class(node_type_id: str):
    if node_type_id == "reasoning_save":
        return ReasoningSaveExecutor
    if node_type_id == "reasoning_load":
        return ReasoningLoadExecutor
    return None