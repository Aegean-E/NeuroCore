from datetime import datetime

class ReasoningBookService:
    def __init__(self):
        self.thoughts = []

    def log_thought(self, content, source="Flow"):
        entry = {
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "content": content,
            "source": source
        }
        # Insert at the beginning (newest first)
        self.thoughts.insert(0, entry)
        # Keep last 100 thoughts
        if len(self.thoughts) > 100:
            self.thoughts.pop()

    def get_thoughts(self):
        return self.thoughts

service = ReasoningBookService()