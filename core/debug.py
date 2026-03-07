from collections import deque
from datetime import datetime
import json
import time
import logging

logger = logging.getLogger(__name__)

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
        # Log to console for immediate feedback using proper logging
        logger.debug(f"[{flow_id}] {node_name} ({event_type}): {json.dumps(details, default=str)}")
    
    def get_logs(self, reverse: bool = True):
        """
        Get all logs.
        
        Args:
            reverse: If True (default), returns newest first (reversed).
                    If False, returns oldest first (chronological).
        
        Returns:
            List of log entries
        """
        if reverse:
            return list(self.logs)[::-1]
        return list(self.logs)
        
    def get_recent_logs(self, since_timestamp=0, reverse: bool = False):
        """
        Get logs since a given timestamp.
        
        Args:
            since_timestamp: Only return logs with timestamp_raw > this value
            reverse: If False (default), returns oldest first (chronological).
                    If True, returns newest first (reversed).
        
        Returns:
            List of log entries in chronological order by default
        """
        result = [log for log in self.logs if log['timestamp_raw'] > float(since_timestamp)]
        if reverse:
            return result[::-1]
        return result

    def clear(self):
        self.logs.clear()

debug_logger = DebugLogger()