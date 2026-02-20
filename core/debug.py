from collections import deque
from datetime import datetime
import json

class DebugLogger:
    def __init__(self, max_logs=50):
        self.logs = deque(maxlen=max_logs)
    
    def log(self, flow_id, node_id, node_name, event_type, details):
        entry = {
            "timestamp": datetime.now().strftime("%H:%M:%S.%f")[:-3],
            "flow_id": flow_id,
            "node_id": node_id,
            "node_name": node_name,
            "event": event_type,
            "details": details
        }
        self.logs.append(entry)
    
    def get_logs(self):
        return list(self.logs)[::-1]

    def clear(self):
        self.logs.clear()

debug_logger = DebugLogger()