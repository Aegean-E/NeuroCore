import json
import os
from datetime import datetime

REASONING_FILE = "data/reasoning_book.json"

class ReasoningBookManager:
    def __init__(self):
        self.file_path = REASONING_FILE
        if not os.path.exists("data"):
            os.makedirs("data", exist_ok=True)
        if not os.path.exists(self.file_path):
            self._save_book([])

    def _load_book(self):
        try:
            with open(self.file_path, 'r') as f:
                return json.load(f)
        except:
            return []

    def _save_book(self, data):
        with open(self.file_path, 'w') as f:
            json.dump(data, f, indent=4)

    def add_entry(self, content, source="Assistant"):
        book = self._load_book()
        entry = {
            "timestamp": datetime.now().isoformat(),
            "source": source,
            "content": content
        }
        book.append(entry)
        # Optional: Keep book size manageable (e.g., last 1000 entries)
        if len(book) > 1000:
            book = book[-1000:]
        self._save_book(book)
    
    def get_context(self, limit=10):
        book = self._load_book()
        recent = book[-limit:]
        if not recent:
            return ""
        
        context_lines = ["### Reasoning Book (Past Context)"]
        for e in recent:
            # Format: [Time] Source: Content
            ts = e.get('timestamp', '')[:19].replace('T', ' ')
            context_lines.append(f"[{ts}] {e.get('content')}")
        context_lines.append("### End Reasoning Book")
        
        return "\n".join(context_lines)

book_manager = ReasoningBookManager()

class ReasoningLoadExecutor:
    async def receive(self, input_data: dict, config: dict = None) -> dict:
        if not isinstance(input_data, dict):
            input_data = {"data": input_data}
            
        limit = int(config.get("limit", 10)) if config else 10
        context = book_manager.get_context(limit)
        
        if context:
            # Inject into messages as a system prompt
            messages = input_data.get("messages", [])
            # We append it to the list. If there are existing system prompts, 
            # this adds another one, which most LLMs handle fine.
            messages.append({"role": "system", "content": context})
            input_data["messages"] = messages
            
        return input_data

    async def send(self, processed_data: dict) -> dict:
        return processed_data

class ReasoningSaveExecutor:
    async def receive(self, input_data: dict, config: dict = None) -> dict:
        # Extract content from various possible input formats (LLM output, raw text, etc.)
        content = None
        
        if isinstance(input_data, dict):
            if "content" in input_data:
                content = input_data["content"]
            elif "choices" in input_data and len(input_data["choices"]) > 0:
                # Standard OpenAI response format
                content = input_data["choices"][0]["message"]["content"]
            elif "messages" in input_data and input_data["messages"]:
                # Try to grab the last assistant message
                last_msg = input_data["messages"][-1]
                if last_msg.get("role") == "assistant":
                    content = last_msg.get("content")
        elif isinstance(input_data, str):
            content = input_data
            
        if content:
            book_manager.add_entry(content)
            
        return input_data

    async def send(self, processed_data: dict) -> dict:
        return processed_data

async def get_executor_class(node_type_id: str):
    if node_type_id == 'reasoning_load':
        return ReasoningLoadExecutor
    elif node_type_id == 'reasoning_save':
        return ReasoningSaveExecutor
    return None