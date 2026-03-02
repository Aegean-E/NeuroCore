import json
import os
import uuid
from datetime import datetime

EVENTS_FILE = "calendar_events.json"

class EventManager:
    def __init__(self, storage_file=EVENTS_FILE):
        self.storage_file = storage_file
        self._ensure_file()

    def _ensure_file(self):
        if not os.path.exists(self.storage_file):
            with open(self.storage_file, "w") as f:
                json.dump([], f)

    def _load_events(self):
        try:
            with open(self.storage_file, "r") as f:
                return json.load(f)
        except:
            return []

    def _save_events(self, events):
        with open(self.storage_file, "w") as f:
            json.dump(events, f, indent=4)

    def add_event(self, title, start_time):
        events = self._load_events()
        event = {
            "id": str(uuid.uuid4()),
            "title": title,
            "start_time": start_time,
            "created_at": datetime.now().isoformat(),
            "notified": False
        }
        events.append(event)
        self._save_events(events)
        return event

    def get_events_by_date(self, date_str):
        events = self._load_events()
        return [e for e in events if e.get("start_time", "").startswith(date_str)]

    def get_upcoming(self, limit=10):
        events = self._load_events()
        now = datetime.now().replace(second=0, microsecond=0)

        def _parse_dt(e):
            try:
                return datetime.strptime(e.get("start_time", "")[:16], "%Y-%m-%d %H:%M")
            except (ValueError, TypeError):
                return datetime.max

        upcoming = [e for e in events if _parse_dt(e) >= now]
        upcoming.sort(key=_parse_dt)
        return upcoming[:limit]

    def get_event_by_id(self, event_id):
        events = self._load_events()
        for e in events:
            if e.get("id") == event_id:
                return e
        return None

    def update_event(self, event_id, title=None, start_time=None):
        """Update an existing event's title and/or start_time. Returns the updated event or None."""
        events = self._load_events()
        for event in events:
            if event.get("id") == event_id:
                if title is not None:
                    event["title"] = title
                if start_time is not None:
                    event["start_time"] = start_time
                self._save_events(events)
                return event
        return None

    def delete_event(self, event_id):
        events = self._load_events()
        initial_count = len(events)
        events = [e for e in events if e.get("id") != event_id]
        self._save_events(events)
        return len(events) < initial_count

event_manager = EventManager()