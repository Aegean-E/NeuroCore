import json
import os
from datetime import datetime

EVENTS_FILE = "calendar_events.json"

class CalendarWatcherExecutor:
    async def receive(self, input_data: dict, config: dict = None) -> dict:
        """
        Checks for calendar events scheduled for the current time.
        Returns None if no events are found, stopping the flow.
        """
        if not os.path.exists(EVENTS_FILE):
            return None
            
        try:
            with open(EVENTS_FILE, "r") as f:
                events = json.load(f)
        except Exception:
            return None

        now = datetime.now()
        triggered_events = []
        
        for event in events:
            try:
                # Expected format: "date": "YYYY-MM-DD", "time": "HH:MM"
                if "date" not in event or "time" not in event:
                    continue
                    
                event_dt_str = f"{event['date']} {event['time']}"
                event_dt = datetime.strptime(event_dt_str, "%Y-%m-%d %H:%M")
                
                # Check if the event is within the current minute
                # We compare minutes to avoid second-level precision issues and double firing
                if event_dt.date() == now.date() and event_dt.hour == now.hour and event_dt.minute == now.minute:
                    triggered_events.append(event)
            except ValueError:
                continue
        
        if not triggered_events:
            # No events found for this minute. Return None to stop the flow.
            # The Repeater node will trigger this again later.
            return None
            
        # Events found! Construct the output.
        titles = [e.get("title", "Untitled Event") for e in triggered_events]
        message = f"📅 Calendar Alert: {', '.join(titles)}"
        
        return {
            "content": message,
            "events": triggered_events,
            "event_count": len(triggered_events)
        }

    async def send(self, processed_data: dict) -> dict:
        return processed_data

async def get_executor_class(node_type_id: str):
    if node_type_id == "calendar_watcher":
        return CalendarWatcherExecutor
    return None