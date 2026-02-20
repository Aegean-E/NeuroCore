import json
import os
import threading
from datetime import datetime

EVENTS_FILE = "calendar_events.json"

class EventManager:
    def __init__(self, storage_file=EVENTS_FILE):
        self.storage_file = storage_file
        self.lock = threading.Lock()
        self.events = self._load_events()

    def _load_events(self):
        if os.path.exists(self.storage_file):
            with open(self.storage_file, "r") as f:
                try:
                    return json.load(f)
                except json.JSONDecodeError:
                    return []
        return []

    def _save_events(self):
        with open(self.storage_file, "w") as f:
            json.dump(self.events, f, indent=4)

    def add_event(self, title, start_time, end_time=None):
        with self.lock:
            event = {
                "id": len(self.events) + 1,
                "title": title,
                "start_time": start_time,
                "end_time": end_time,
                "created_at": datetime.now().isoformat()
            }
            self.events.append(event)
            # Sort by start time
            self.events.sort(key=lambda x: x.get("start_time", ""))
            self._save_events()
            return event

    def get_upcoming(self, limit=5):
        # Simple filter for future events could be added here
        return self.events[:limit]
        
    def get_events_by_date(self, date_str):
        # date_str expected as YYYY-MM-DD
        results = []
        for event in self.events:
            if event.get("start_time", "").startswith(date_str):
                results.append(event)
        return results
        
    def delete_event(self, event_id):
        with self.lock:
            try:
                event_id = int(event_id)
            except ValueError:
                return False
            
            self.events = [e for e in self.events if e.get("id") != event_id]
            self._save_events()
            return True

event_manager = EventManager()