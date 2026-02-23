import os
import json
from datetime import datetime

DATA_FILE = "data/reasoning_book.json"
MAX_THOUGHTS = 100

class ReasoningBookService:
    def __init__(self):
        self.thoughts = []
        self._load()

    def _load(self):
        if os.path.exists(DATA_FILE):
            try:
                with open(DATA_FILE, "r") as f:
                    self.thoughts = json.load(f)
            except Exception:
                self.thoughts = []

    def _save(self):
        os.makedirs(os.path.dirname(DATA_FILE) or "data", exist_ok=True)
        try:
            with open(DATA_FILE, "w") as f:
                json.dump(self.thoughts, f)
        except Exception:
            pass

    def log_thought(self, content, source="Flow"):
        entry = {
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "content": content,
            "source": source
        }
        self.thoughts.insert(0, entry)
        if len(self.thoughts) > MAX_THOUGHTS:
            self.thoughts = self.thoughts[:MAX_THOUGHTS]
        self._save()

    def get_thoughts(self):
        return self.thoughts

    def clear(self):
        self.thoughts = []
        self._save()

service = ReasoningBookService()