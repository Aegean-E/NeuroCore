from .service import service

class LogThoughtExecutor:
    async def receive(self, data: dict, config: dict = None) -> dict:
        config = config or {}
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

async def get_executor_class(node_type_id: str):
    if node_type_id == "log_thought":
        return LogThoughtExecutor
    return None