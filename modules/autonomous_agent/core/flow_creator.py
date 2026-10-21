"""
Flow Creator - Agent creates and manages its own flows dynamically.
"""

import asyncio
import json
import uuid
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime

from .types import FlowSpec, AgentConfig, AgentAction, ActionType
from .sandbox import AgentSandbox, SandboxViolation, FlowValidationError
from .memory import AgentMemory


# Built-in node templates the agent can use
NODE_TEMPLATES = {
    "chat_input": {
        "moduleId": "chat",
        "nodeTypeId": "chat_input",
        "config": {}
    },
    "chat_output": {
        "moduleId": "chat",
        "nodeTypeId": "chat_output",
        "config": {}
    },
    "llm_module": {
        "moduleId": "llm_module",
        "nodeTypeId": "llm_module",
        "config": {}
    },
    "system_prompt": {
        "moduleId": "system_prompt",
        "nodeTypeId": "system_prompt",
        "config": {"system_prompt": ""}
    },
    "memory_recall": {
        "moduleId": "memory",
        "nodeTypeId": "memory_recall",
        "config": {"limit": 5, "threshold": 0.7}
    },
    "memory_save": {
        "moduleId": "memory",
        "nodeTypeId": "memory_save",
        "config": {}
    },
    "tool_dispatcher": {
        "moduleId": "tools",
        "nodeTypeId": "tool_dispatcher",
        "config": {"allowed_tools": ["Calculator", "TimeZoneConverter", "SystemTime"]}
    },
    "conditional_router": {
        "moduleId": "logic",
        "nodeTypeId": "conditional_router",
        "config": {
            "check_field": "tool_calls",
            "true_branches": [],
            "false_branches": []
        }
    },
    "repeater_node": {
        "moduleId": "logic",
        "nodeTypeId": "repeater_node",
        "config": {"delay": 300, "max_repeats": 0}
    },
    "calendar_watcher": {
        "moduleId": "calendar",
        "nodeTypeId": "calendar_watcher",
        "config": {}
    },
    "telegram_output": {
        "moduleId": "messaging_bridge",
        "nodeTypeId": "telegram_output",
        "config": {}
    },
    "kb_query": {
        "moduleId": "knowledge_base",
        "nodeTypeId": "kb_query",
        "config": {"limit": 5, "threshold": 0.6}
    }
}

# Common flow patterns
FLOW_PATTERNS = {
    "simple_chat": {
        "nodes": ["chat_input", "llm_module", "chat_output"],
        "connections": [
            {"from": "chat_input", "to": "llm_module"},
            {"from": "llm_module", "to": "chat_output"}
        ]
    },
    "memory_chat": {
        "nodes": ["chat_input", "memory_recall", "system_prompt", "llm_module", "memory_save", "chat_output"],
        "connections": [
            {"from": "chat_input", "to": "memory_recall"},
            {"from": "memory_recall", "to": "system_prompt"},
            {"from": "system_prompt", "to": "llm_module"},
            {"from": "llm_module", "to": "memory_save"},
            {"from": "llm_module", "to": "chat_output"}
        ]
    },
    "tool_agent": {
        "nodes": ["chat_input", "system_prompt", "llm_module", "conditional_router", "tool_dispatcher", "llm_module", "chat_output"],
        "connections": [
            {"from": "chat_input", "to": "system_prompt"},
            {"from": "system_prompt", "to": "llm_module"},
            {"from": "llm_module", "to": "conditional_router"},
            {"from": "conditional_router", "to": "tool_dispatcher"},
            {"from": "tool_dispatcher", "to": "llm_module"},
            {"from": "conditional_router", "to": "chat_output"}
        ],
        "reverted": ["tool_dispatcher"]
    },
    "scheduled_notification": {
        "nodes": ["repeater_node", "kb_query", "system_prompt", "llm_module", "telegram_output"],
        "connections": [
            {"from": "repeater_node", "to": "kb_query"},
            {"from": "kb_query", "to": "system_prompt"},
            {"from": "system_prompt", "to": "llm_module"},
            {"from": "llm_module", "to": "telegram_output"}
        ]
    },
    "calendar_triggered": {
        "nodes": ["calendar_watcher", "system_prompt", "llm_module", "telegram_output"],
        "connections": [
            {"from": "calendar_watcher", "to": "system_prompt"},
            {"from": "system_prompt", "to": "llm_module"},
            {"from": "llm_module", "to": "telegram_output"}
        ]
    }
}


class FlowCreator:
    """Agent creates and manages its own flows dynamically."""
    
    def __init__(self, agent_id: str, sandbox: AgentSandbox, memory: AgentMemory, config: AgentConfig):
        self.agent_id = agent_id
        self.sandbox = sandbox
        self.memory = memory
        self.config = config
        self.created_flows: Dict[str, Dict[str, Any]] = {}
    
    async def create_flow(self, specification: FlowSpec) -> Dict[str, Any]:
        """Create a new flow from specification."""
        # 1. Validate against sandbox limits
        self.sandbox.validate_flow_spec(specification)
        
        # 2. Generate flow from pattern or custom
        flow_json = await self._generate_flow(specification)
        
        # 3. Validate flow structure
        validation = await self._validate_flow(flow_json)
        if not validation["valid"]:
            raise FlowValidationError(f"Flow validation failed: {validation['errors']}")
        
        # 4. Save with ownership metadata
        flow_id = f"agent_{self.agent_id}_{uuid.uuid4().hex[:8]}"
        flow_json["id"] = flow_id
        flow_json["name"] = specification.description[:100]
        flow_json["metadata"] = {
            "creator": self.agent_id,
            "created_at": datetime.now().isoformat(),
            "specification": specification.to_dict(),
            "version": 1
        }
        
        # 5. Register ownership
        self.sandbox.register_flow_ownership(flow_id)
        self.created_flows[flow_id] = flow_json
        
        return flow_json
    
    async def _generate_flow(self, spec: FlowSpec) -> Dict[str, Any]:
        """Generate flow JSON from specification."""
        # Try to match a known pattern
        pattern = self._match_pattern(spec)
        
        if pattern:
            return await self._build_from_pattern(pattern, spec)
        else:
            return await self._build_custom_flow(spec)
    
    def _match_pattern(self, spec: FlowSpec) -> Optional[str]:
        """Match specification to a known flow pattern."""
        desc = spec.description.lower()
        
        if any(kw in desc for kw in ["chat", "conversation", "talk"]):
            if "memory" in desc or "remember" in desc:
                return "memory_chat"
            return "simple_chat"
        
        if any(kw in desc for kw in ["tool", "function", "calculate", "weather", "fetch"]):
            return "tool_agent"
        
        if any(kw in desc for kw in ["schedule", "every", "daily", "hourly", "repeating"]):
            if "telegram" in desc or "notify" in desc:
                return "scheduled_notification"
            return "repeater_node"
        
        if any(kw in desc for kw in ["calendar", "event", "meeting", "appointment"]):
            return "calendar_triggered"
        
        return None
    
    async def _build_from_pattern(self, pattern_name: str, spec: FlowSpec) -> Dict[str, Any]:
        """Build flow from predefined pattern."""
        pattern = FLOW_PATTERNS[pattern_name]
        
        nodes = []
        for i, node_type in enumerate(pattern["nodes"]):
            template = NODE_TEMPLATES[node_type].copy()
            node_id = f"node-{i}"
            node = {
                "id": node_id,
                "nodeTypeId": template["nodeTypeId"],
                "moduleId": template["moduleId"],
                "name": template["nodeTypeId"].replace("_", " ").title(),
                "config": template["config"].copy()
            }
            
            # Customize based on spec
            if node_type == "system_prompt" and spec.expected_output:
                node["config"]["system_prompt"] = f"Task: {spec.description}\nExpected output: {spec.expected_output}"
            elif node_type == "tool_dispatcher":
                allowed = spec.metadata.get("allowed_tools", ["Calculator", "TimeZoneConverter", "SystemTime"])
                node["config"]["allowed_tools"] = allowed
            elif node_type == "repeater_node":
                delay = spec.metadata.get("delay_seconds", 300)
                node["config"]["delay"] = delay
            
            nodes.append(node)
        
        # Build connections
        connections = []
        for conn in pattern["connections"]:
            # Map node names to IDs
            from_idx = pattern["nodes"].index(conn["from"])
            to_idx = pattern["nodes"].index(conn["to"])
            connections.append({
                "from": f"node-{from_idx}",
                "to": f"node-{to_idx}"
            })
        
        # Handle reverted nodes
        reverted = pattern.get("reverted", [])
        
        return {
            "nodes": nodes,
            "connections": connections,
            "bridges": [],
            "reverted_nodes": reverted
        }
    
    async def _build_custom_flow(self, spec: FlowSpec) -> Dict[str, Any]:
        """Build custom flow using LLM (delegated to sandbox)."""
        # This would use the agent's LLM access through sandbox
        # For now, return a basic template
        return await self._build_from_pattern("simple_chat", spec)
    
    async def _validate_flow(self, flow_json: Dict[str, Any]) -> Dict[str, Any]:
        """Validate flow structure."""
        errors = []
        
        # Basic structure checks
        if not flow_json.get("nodes"):
            errors.append("Flow must have at least one node")
        
        if not flow_json.get("connections"):
            errors.append("Flow must have connections")
        
        # Check node IDs are unique
        node_ids = [n["id"] for n in flow_json.get("nodes", [])]
        if len(node_ids) != len(set(node_ids)):
            errors.append("Duplicate node IDs")
        
        # Check connections reference valid nodes
        valid_ids = set(node_ids)
        for conn in flow_json.get("connections", []):
            if conn["from"] not in valid_ids:
                errors.append(f"Connection from unknown node: {conn['from']}")
            if conn["to"] not in valid_ids:
                errors.append(f"Connection to unknown node: {conn['to']}")
        
        return {"valid": len(errors) == 0, "errors": errors}
    
    async def update_flow(self, flow_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Update an existing flow."""
        if not self.sandbox.check_flow_ownership(flow_id):
            raise SandboxViolation("Cannot modify flow not owned by agent")
        
        if flow_id not in self.created_flows:
            raise ValueError(f"Flow {flow_id} not found in agent's flows")
        
        flow = self.created_flows[flow_id].copy()
        flow.update(updates)
        flow["metadata"]["version"] = flow["metadata"].get("version", 1) + 1
        flow["metadata"]["updated_at"] = datetime.now().isoformat()
        
        # Validate updated flow
        validation = await self._validate_flow(flow)
        if not validation["valid"]:
            raise FlowValidationError(f"Updated flow invalid: {validation['errors']}")
        
        self.created_flows[flow_id] = flow
        return flow
    
    async def delete_flow(self, flow_id: str) -> bool:
        """Delete a flow."""
        if not self.sandbox.check_flow_ownership(flow_id):
            raise SandboxViolation("Cannot delete flow not owned by agent")
        
        if flow_id in self.created_flows:
            del self.created_flows[flow_id]
            return True
        return False
    
    def get_owned_flows(self) -> List[Dict[str, Any]]:
        """Get list of flows owned by this agent."""
        return [
            {
                "id": fid,
                "name": flow.get("name"),
                "created_at": flow.get("metadata", {}).get("created_at"),
                "version": flow.get("metadata", {}).get("version", 1)
            }
            for fid, flow in self.created_flows.items()
        ]
    
    def get_flow(self, flow_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific flow."""
        return self.created_flows.get(flow_id)