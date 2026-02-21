import json
import os
from datetime import datetime
from core.debug import debug_logger

EVENTS_FILE = "calendar_events.json"

class CalendarWatcherExecutor:
    """
    Checks for calendar events scheduled for the current time.
    """
    async def receive(self, input_data: dict, config: dict = None) -> dict:
        # This node is designed to be triggered periodically (e.g., by a Repeater node).
        # It checks if any events in calendar_events.json match the current minute.
        flow_id = config.get("_flow_id", "unknown")

        if not os.path.exists(EVENTS_FILE):
            return None

        try:
            with open(EVENTS_FILE, "r") as f:
                events = json.load(f)
        except Exception as e:
            # Log to console but return None to stop the flow and prevent Telegram spam
            print(f"[Calendar Watcher] Error loading file: {e}")
            return None

        if not isinstance(events, list):
            return None

        now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
        
        # Log the check to the Debug tab so we know it's running
        debug_logger.log(flow_id, "calendar_watcher", "Calendar Watcher", "info", f"Checking time: {now_str} against {len(events)} events")

        due_events = []
        events_updated = False

        for event in events:
            # Normalize time format (handle T separator if present)
            start_time = event.get("start_time", "").replace("T", " ")
            
            # Check if time is now or in the past, and not yet notified
            if start_time and start_time <= now_str and not event.get("notified", False):
                due_events.append(event)
                event["notified"] = True
                events_updated = True

        if not due_events:
            # Return None to stop the flow execution for this branch
            # debug_logger.log(flow_id, "calendar_watcher", "Calendar Watcher", "info", "No events due.")
            return None

        # Save the updated events back to the file if any were modified
        if events_updated:
            try:
                with open(EVENTS_FILE, "w") as f:
                    json.dump(events, f, indent=4)
            except Exception as e:
                return {"error": f"Failed to update calendar: {str(e)}"}

        debug_logger.log(flow_id, "calendar_watcher", "Calendar Watcher", "success", f"Found {len(due_events)} due events")

        # Format the output message
        messages = [f"🔔 Reminder: {e.get('title', 'Untitled Event')}" for e in due_events]
        return {"content": "\n".join(messages)}

    async def send(self, processed_data: dict) -> dict:
        return processed_data

async def get_executor_class(node_type_id: str):
    if node_type_id == 'calendar_watcher':
        return CalendarWatcherExecutor
    return None