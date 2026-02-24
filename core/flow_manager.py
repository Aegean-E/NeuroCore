import json
import os
import uuid
import copy
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
        
        system_prompt = (
            "You are an **Autonomous Recursive Thinking Agent**.\n"
            "Your goal is to solve complex problems by breaking them down, executing steps, and iterating until the solution is found.\n\n"
            "### 🧠 CONTEXT AWARENESS\n"
            "You have access to the following dynamic context streams. Use them to maintain continuity:\n"
            "1. **Past Reasoning**: Review your previous thoughts to ensure logical progression and avoid loops.\n"
            "2. **Long-Term Memory**: Recall facts and preferences about the project.\n"
            "3. **Knowledge Base**: Use retrieved document snippets to ground your answers in facts.\n\n"
            "### 🔄 OPERATIONAL LOOP\n"
            "Your existence is a continuous loop of thought and action. In each cycle, perform **ONE** of the following:\n\n"
            "**OPTION A: ACTION (Tool Call)**\n"
            "If you need external information (e.g., check time, search web, read file) or need to perform an action:\n"
            "- CALL THE APPROPRIATE TOOL.\n"
            "- Do not output reasoning text if you are calling a tool (unless the tool requires it).\n\n"
            "**OPTION B: REASONING (Internal Thought)**\n"
            "If you have enough information to proceed or need to analyze data:\n"
            "1. **Synthesize**: Combine new tool outputs with your Past Reasoning.\n"
            "2. **Plan**: Determine the immediate next step.\n"
            "3. **Output**: Write a clear, concise reasoning entry. This text will be saved to your reasoning history for the next cycle.\n\n"
            "### 🛑 CRITICAL RULES\n"
            "- **AUTONOMY**: Do not ask the user for input or clarification. If information is missing, use tools to find it or make a reasonable assumption to proceed.\n"
            "- **NO REPETITION**: Check Past Reasoning. If you have already tried something that failed, try a different approach.\n"
            "- **INCREMENTAL PROGRESS**: Do not try to solve everything at once. Take one logical step per cycle.\n"
            "- **TERMINATION**: When the objective is fully satisfied, output a final summary and explicitly state \"TASK COMPLETED\"."
        )

        return {
            flow_id: {
                "id": flow_id,
                "name": "Default Chat Flow",
                "nodes": [
                    {"id": "node-0", "moduleId": "chat", "nodeTypeId": "chat_input", "name": "Chat Input", "x": -97, "y": 248, "config": {}, "isReverted": False, "outputDot": {}, "inputDot": {}},
                    {"id": "node-1", "moduleId": "system_prompt", "nodeTypeId": "system_prompt", "name": "System Prompt", "x": 343.9, "y": 203.9, "config": {"system_prompt": system_prompt, "enabled_tools": ["Weather", "Calculator", "TimeZoneConverter", "ConversionCalculator", "SystemTime", "FetchURL", "CurrencyConverter", "SaveReminder", "CheckCalendar"], "explanation": ""}, "isReverted": False, "outputDot": {}, "inputDot": {}},
                    {"id": "node-2", "moduleId": "llm_module", "nodeTypeId": "llm_module", "name": "LLM Core", "x": 551, "y": 250, "config": {}, "isReverted": False, "outputDot": {}, "inputDot": {}},
                    {"id": "node-3", "moduleId": "chat", "nodeTypeId": "chat_output", "name": "Chat Output", "x": 908, "y": 250, "config": {}, "isReverted": False, "outputDot": {}, "inputDot": {}},
                    {"id": "node-4", "moduleId": "memory", "nodeTypeId": "memory_save", "name": "Memory Save", "x": 123, "y": 163, "config": {}, "isReverted": False, "outputDot": {}, "inputDot": {}},
                    {"id": "node-5", "moduleId": "memory", "nodeTypeId": "memory_save", "name": "Memory Save", "x": 1134, "y": 249, "config": {}, "isReverted": False, "outputDot": {}, "inputDot": {}},
                    {"id": "node-6", "moduleId": "memory", "nodeTypeId": "memory_recall", "name": "Memory Recall", "x": 123, "y": 249, "config": {}, "isReverted": False, "outputDot": {}, "inputDot": {}},
                    {"id": "node-7", "moduleId": "telegram", "nodeTypeId": "telegram_output", "name": "Telegram Output", "x": 909, "y": 337, "config": {}, "isReverted": False, "outputDot": {}, "inputDot": {}},
                    {"id": "node-8", "moduleId": "telegram", "nodeTypeId": "telegram_input", "name": "Telegram Input", "x": -289, "y": 249, "config": {}, "isReverted": False, "outputDot": {}, "inputDot": {}},
                    {"id": "node-9", "moduleId": "tools", "nodeTypeId": "tool_dispatcher", "name": "Tool Dispatcher", "x": 632.1, "y": -28.2, "config": {"allowed_tools": ["Weather", "Calculator", "TimeZoneConverter", "ConversionCalculator", "SystemTime", "FetchURL", "CurrencyConverter", "SaveReminder", "CheckCalendar"], "explanation": ""}, "isReverted": True, "outputDot": {}, "inputDot": {}},
                    {"id": "node-10", "moduleId": "logic", "nodeTypeId": "conditional_router", "name": "Conditional Router", "x": 717, "y": 250, "config": {"check_field": "tool_calls", "true_branches": ["node-9"], "false_branches": ["node-3", "node-7"], "explanation": ""}, "isReverted": False, "outputDot": {}, "inputDot": {}},
                    {"id": "node-11", "moduleId": "calendar", "nodeTypeId": "calendar_watcher", "name": "Calendar Watcher", "x": 719.2, "y": 336.8, "config": {}, "isReverted": False, "outputDot": {}, "inputDot": {}},
                    {"id": "node-12", "moduleId": "logic", "nodeTypeId": "repeater_node", "name": "Repeater", "x": 549.9, "y": 336.0, "config": {"delay": 30, "max_repeats": 0, "explanation": ""}, "isReverted": False, "outputDot": {}, "inputDot": {}}
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
                    {"from": "node-10", "to": "node-9"},
                    {"from": "node-11", "to": "node-7"},
                    {"from": "node-12", "to": "node-11"}
                ],
                "bridges": [],
                "created_at": datetime.now().isoformat()
            }
        }

    def _ensure_default_active(self):
        if not settings.get("active_ai_flows"):
            settings.save_settings({"active_ai_flows": []})

    def _save_flows_to_disk(self, flows):
        # Lock should be held by caller
        with open(self.storage_file, "w") as f:
            json.dump(flows, f, indent=4)

    def _save_flows(self):
        self._save_flows_to_disk(self.flows)

    def save_flow(self, name, nodes, connections, bridges=None, flow_id=None):
        with self.lock:
            if flow_id is None:
                flow_id = str(uuid.uuid4())
            
            self.flows[flow_id] = {
                "id": flow_id,
                "name": name,
                "nodes": nodes,
                "connections": connections,
                "bridges": bridges or [],
                "created_at": datetime.now().isoformat()
            }
            self._save_flows()
            return self.flows[flow_id]

    def import_flows(self, flows_data: dict):
        """Replaces all flows with the provided dictionary."""
        with self.lock:
            self.flows = flows_data
            self._save_flows()
            self._ensure_default_active()

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

    def make_active_flow_default(self):
        """Overwrites the default flow with the first active flow."""
        with self.lock:
            active_ids = settings.get("active_ai_flows", [])
            if not active_ids or active_ids[0] not in self.flows:
                return False
            
            active_id = active_ids[0]
            active_flow = copy.deepcopy(self.flows[active_id])
            
            default_id = "default-flow-001"
            active_flow["id"] = default_id
            active_flow["name"] = "Default Chat Flow"
            active_flow["created_at"] = datetime.now().isoformat()
            
            self.flows[default_id] = active_flow
            self._save_flows()
            return True

flow_manager = FlowManager()