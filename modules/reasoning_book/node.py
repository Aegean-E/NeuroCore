from .service import service

MIN_CONTENT_LENGTH = 30


class ReasoningSaveExecutor:
    async def receive(self, data: dict, config: dict = None) -> dict:
        config = config or {}

        # Skip logging if there's an error in the data
        if isinstance(data, dict) and data.get("error"):
            return data

        content = None

        if isinstance(data, dict):
            # Check configurable source field first (default: "content" — matches agent_loop output)
            source_field = config.get("source_field", "content")
            content = data.get(source_field)

            # Fall back to other common reasoning fields if source_field not found
            if not content:
                for field in ("reasoning", "thought", "summary", "conclusion", "result"):
                    if field in data:
                        content = data[field]
                        break

        # If no meaningful content found, skip
        if not content:
            return data

        # Skip if too short
        if len(str(content)) < MIN_CONTENT_LENGTH:
            return data

        service.log_thought(str(content), source=config.get("source", "Flow Node"))
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