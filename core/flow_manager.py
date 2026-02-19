import json
import os
import uuid
from datetime import datetime

FLOWS_FILE = "ai_flows.json"

class FlowManager:
    def __init__(self, storage_file=FLOWS_FILE):
        self.storage_file = storage_file
        self.flows = self._load_flows()

    def _load_flows(self):
        if not os.path.exists(self.storage_file):
            with open(self.storage_file, "w") as f:
                json.dump({}, f, indent=4)
            return {}
        
        with open(self.storage_file, "r") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {}

    def _save_flows(self):
        with open(self.storage_file, "w") as f:
            json.dump(self.flows, f, indent=4)

    def save_flow(self, name, nodes, connections, flow_id=None):
        if flow_id is None:
            flow_id = str(uuid.uuid4())
        
        self.flows[flow_id] = {
            "id": flow_id,
            "name": name,
            "nodes": nodes,
            "connections": connections,
            "created_at": datetime.now().isoformat()
        }
        self._save_flows()
        return self.flows[flow_id]

    def get_flow(self, flow_id):
        return self.flows.get(flow_id)

    def list_flows(self):
        return sorted(self.flows.values(), key=lambda x: x.get('created_at', ''), reverse=True)

    def delete_flow(self, flow_id):
        if flow_id in self.flows:
            del self.flows[flow_id]
            self._save_flows()
            return True
        return False

flow_manager = FlowManager()