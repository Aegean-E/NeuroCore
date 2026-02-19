import json
import os
import uuid
from datetime import datetime
from core.settings import settings

FLOWS_FILE = "ai_flows.json"

class FlowManager:
    def __init__(self, storage_file=FLOWS_FILE):
        self.storage_file = storage_file
        self.flows = self._load_flows()

    def _load_flows(self):
        if not os.path.exists(self.storage_file):
            default_flows = self._create_default_flows()
            self._save_flows_to_disk(default_flows)
            self._ensure_default_active()
            return default_flows
        
        with open(self.storage_file, "r") as f:
            try:
                flows = json.load(f)
                if not flows:
                    flows = self._create_default_flows()
                    self._save_flows_to_disk(flows)
                    self._ensure_default_active()
                return flows
            except json.JSONDecodeError:
                default_flows = self._create_default_flows()
                self._save_flows_to_disk(default_flows)
                self._ensure_default_active()
                return default_flows

    def _create_default_flows(self):
        flow_id = "default-flow-001"
        return {
            flow_id: {
                "id": flow_id,
                "name": "Default Chat Flow",
                "nodes": [
                    {"id": "node-0", "moduleId": "chat", "nodeTypeId": "chat_input", "name": "Chat Input", "x": -97, "y": 248, "config": {}},
                    {"id": "node-1", "moduleId": "system_prompt", "nodeTypeId": "system_prompt", "name": "System Prompt", "x": 336, "y": 249, "config": {"system_prompt": "You are NeuroCore, a helpful and intelligent AI assistant."}},
                    {"id": "node-2", "moduleId": "llm_module", "nodeTypeId": "llm_module", "name": "LLM Core", "x": 550, "y": 250, "config": {}},
                    {"id": "node-3", "moduleId": "chat", "nodeTypeId": "chat_output", "name": "Chat Output", "x": 765, "y": 250, "config": {}},
                    {"id": "node-4", "moduleId": "memory", "nodeTypeId": "memory_save", "name": "Memory Save", "x": 113, "y": 165, "config": {}},
                    {"id": "node-5", "moduleId": "memory", "nodeTypeId": "memory_save", "name": "Memory Save", "x": 979, "y": 250, "config": {}},
                    {"id": "node-6", "moduleId": "memory", "nodeTypeId": "memory_recall", "name": "Memory Recall", "x": 112, "y": 249, "config": {}}
                ],
                "connections": [
                    {"from": "node-1", "to": "node-2"},
                    {"from": "node-2", "to": "node-3"},
                    {"from": "node-0", "to": "node-4"},
                    {"from": "node-3", "to": "node-5"},
                    {"from": "node-0", "to": "node-6"},
                    {"from": "node-6", "to": "node-1"}
                ],
                "created_at": datetime.now().isoformat()
            }
        }

    def _ensure_default_active(self):
        if not settings.get("active_ai_flow"):
            settings.save_settings({"active_ai_flow": "default-flow-001"})

    def _save_flows_to_disk(self, flows):
        with open(self.storage_file, "w") as f:
            json.dump(flows, f, indent=4)

    def _save_flows(self):
        self._save_flows_to_disk(self.flows)

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

    def rename_flow(self, flow_id, new_name):
        if flow_id in self.flows:
            self.flows[flow_id]["name"] = new_name
            self._save_flows()
            return True
        return False

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