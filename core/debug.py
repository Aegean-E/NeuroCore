from collections import deque
from datetime import datetime
import json
import time

class DebugLogger:
    def __init__(self, max_logs=50):
        self.logs = deque(maxlen=max_logs)
    
    def log(self, flow_id, node_id, node_name, event_type, details):
        entry = {
            "timestamp": datetime.now().strftime("%H:%M:%S.%f")[:-3],
            "timestamp_raw": time.time(),
            "flow_id": flow_id,
            "node_id": node_id,
            "node_name": node_name,
            "event": event_type,
            "details": details
        }
        self.logs.append(entry)
        # Print to console for immediate feedback
        print(f"[DEBUG] [{flow_id}] {node_name} ({event_type}): {json.dumps(details, default=str)}")
    
    def get_logs(self):
        return list(self.logs)[::-1]
        
    def get_recent_logs(self, since_timestamp=0):
        return [log for log in self.logs if log['timestamp_raw'] > float(since_timestamp)]

    def clear(self):
        self.logs.clear()

debug_logger = DebugLogger()