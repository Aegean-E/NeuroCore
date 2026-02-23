from .service import service

class ReasoningSaveExecutor:
    async def receive(self, data: dict, config: dict = None) -> dict:
        config = config or {}
        
        # Skip logging if there's an error in the data
        if isinstance(data, dict) and data.get("error"):
            return data
        
        content = "No content"
        
        if isinstance(data, dict):
            # Try to find meaningful content fields
            if "reasoning" in data:
                content = data["reasoning"]
            elif "thought" in data:
                content = data["thought"]
            elif "content" in data:
                content = data["content"]
            else:
                content = str(data)
        elif data is not None:
            content = str(data)
            
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