import json
import os
import uuid
from datetime import datetime
import threading
from core.settings import settings

FLOWS_FILE = "ai_flows.json"

class FlowManager:
    def __init__(self, storage_file=FLOWS_FILE):
        self.storage_file = storage_file
        self.lock = threading.Lock()
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
                    {"id": "node-0", "moduleId": "chat", "nodeTypeId": "chat_input", "name": "Chat Input", "x": -97, "y": 248, "config": {}, "isReverted": False, "outputDot": {}, "inputDot": {}},
                    {"id": "node-1", "moduleId": "system_prompt", "nodeTypeId": "system_prompt", "name": "System Prompt", "x": 346, "y": 205, "config": {"system_prompt": "You are NeuroCore, a helpful and intelligent AI assistant.", "enabled_tools": ["Weather"], "explanation": ""}, "isReverted": False, "outputDot": {}, "inputDot": {}},
                    {"id": "node-2", "moduleId": "llm_module", "nodeTypeId": "llm_module", "name": "LLM Core", "x": 551, "y": 250, "config": {}, "isReverted": False, "outputDot": {}, "inputDot": {}},
                    {"id": "node-3", "moduleId": "chat", "nodeTypeId": "chat_output", "name": "Chat Output", "x": 908, "y": 250, "config": {}, "isReverted": False, "outputDot": {}, "inputDot": {}},
                    {"id": "node-4", "moduleId": "memory", "nodeTypeId": "memory_save", "name": "Memory Save", "x": 123, "y": 163, "config": {}, "isReverted": False, "outputDot": {}, "inputDot": {}},
                    {"id": "node-5", "moduleId": "memory", "nodeTypeId": "memory_save", "name": "Memory Save", "x": 1134, "y": 249, "config": {}, "isReverted": False, "outputDot": {}, "inputDot": {}},
                    {"id": "node-6", "moduleId": "memory", "nodeTypeId": "memory_recall", "name": "Memory Recall", "x": 123, "y": 249, "config": {}, "isReverted": False, "outputDot": {}, "inputDot": {}},
                    {"id": "node-7", "moduleId": "telegram", "nodeTypeId": "telegram_output", "name": "Telegram Output", "x": 909, "y": 337, "config": {}, "isReverted": False, "outputDot": {}, "inputDot": {}},
                    {"id": "node-8", "moduleId": "telegram", "nodeTypeId": "telegram_input", "name": "Telegram Input", "x": -289, "y": 249, "config": {}, "isReverted": False, "outputDot": {}, "inputDot": {}},
                    {"id": "node-9", "moduleId": "tools", "nodeTypeId": "tool_dispatcher", "name": "Tool Dispatcher", "x": 627, "y": 144, "config": {"allowed_tools": ["Weather"], "explanation": ""}, "isReverted": True, "outputDot": {}, "inputDot": {}},
                    {"id": "node-10", "moduleId": "logic", "nodeTypeId": "conditional_router", "name": "Conditional Router", "x": 717, "y": 250, "config": {"condition_type": "tool_exists", "check_field": "tool_results", "invert": False, "true_branches": ["node-9"], "false_branches": ["node-3", "node-7"], "explanation": ""}, "isReverted": False, "outputDot": {}, "inputDot": {}}
                ],
                "connections": [
                    {"from": "node-1", "to": "node-2"},
                    {"from": "node-0", "to": "node-4"},
                    {"from": "node-3", "to": "node-5"},
                    {"from": "node-0", "to": "node-6"},
                    {"from": "node-6", "to": "node-1"},
                    {"from": "node-8", "to": "node-0"},
                    {"from": "node-9", "to": "node-2"},
                    {"from": "node-10", "to": "node-3"},
                    {"from": "node-10", "to": "node-7"},
                    {"from": "node-2", "to": "node-10"},
                    {"from": "node-10", "to": "node-9"}
                ],
                "created_at": datetime.now().isoformat()
            }
        }

    def _ensure_default_active(self):
        if not settings.get("active_ai_flow"):
            settings.save_settings({"active_ai_flow": "default-flow-001"})

    def _save_flows_to_disk(self, flows):
        # Lock should be held by caller
        with open(self.storage_file, "w") as f:
            json.dump(flows, f, indent=4)

    def _save_flows(self):
        self._save_flows_to_disk(self.flows)

    def save_flow(self, name, nodes, connections, flow_id=None):
        with self.lock:
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
        with self.lock:
            if flow_id in self.flows:
                self.flows[flow_id]["name"] = new_name
                self._save_flows()
                return True
            return False

    def get_flow(self, flow_id):
        with self.lock:
            return self.flows.get(flow_id)

    def list_flows(self):
        with self.lock:
            return sorted(self.flows.values(), key=lambda x: x.get('created_at', ''), reverse=True)

    def delete_flow(self, flow_id):
        with self.lock:
            if flow_id in self.flows:
                del self.flows[flow_id]
                self._save_flows()
                return True
            return False

flow_manager = FlowManager()