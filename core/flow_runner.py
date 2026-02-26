import asyncio
import importlib
import sys
import types
from collections import deque
from .flow_manager import flow_manager
from core.settings import settings
from core.debug import debug_logger

class FlowRunner:
    _executor_cache = {}

    @classmethod
    def clear_cache(cls):
        cls._executor_cache.clear()

    def __init__(self, flow_id: str, flow_override: dict = None):
        self.flow_id = flow_id
        if flow_override:
            self.flow = flow_override
        else:
            self.flow = flow_manager.get_flow(flow_id)
            
        if not self.flow:
            raise ValueError(f"Flow with id {flow_id} not found.")
        
        self.nodes = {node['id']: node for node in self.flow['nodes']}
        self.connections = self.flow['connections']
        self.bridges = self.flow.get('bridges', [])
        self.bridge_groups = self._build_bridge_groups()
        self.execution_order = self._compute_execution_order()
        
        if settings.get("debug_mode"):
            print(f"[FlowRunner] Initialized for flow {flow_id}")

    def _build_bridge_groups(self):
        """Identifies groups of bridged nodes."""
        adj = {node_id: [] for node_id in self.nodes}
        for b in self.bridges:
            if b['from'] in adj and b['to'] in adj:
                adj[b['from']].append(b['to'])
                adj[b['to']].append(b['from'])
        
        groups = {}
        visited = set()
        for node_id in self.nodes:
            if node_id in visited: continue
            if not adj[node_id]: continue
            
            # BFS to find connected component
            component = []
            queue = deque([node_id])
            visited.add(node_id)
            while queue:
                curr = queue.popleft()
                component.append(curr)
                for neighbor in adj[curr]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            
            for member in component:
                groups[member] = component
        return groups
    
    def _get_bridge_order(self, node_id):
        """Get bridge chain nodes in execution order (upstream to downstream)."""
        if node_id not in self.bridge_groups:
            return []
        
        # Build bridge direction map
        bridge_dir = {}
        for b in self.bridges:
            from_node = b['from']
            to_node = b['to']
            # from_node runs BEFORE to_node
            if from_node not in bridge_dir:
                bridge_dir[from_node] = []
            bridge_dir[from_node].append(to_node)
        
        # Find all nodes in chain starting from furthest upstream
        group = self.bridge_groups[node_id]
        ordered = []
        visited = set()
        
        def visit(nid):
            if nid in visited:
                return
            visited.add(nid)
            # Visit dependencies first
            for b in self.bridges:
                if b['to'] == nid and b['from'] in group:
                    visit(b['from'])
            ordered.append(nid)
        
        for nid in group:
            visit(nid)
        
        return ordered

    def _compute_execution_order(self):
        """Performs a topological sort (Kahn's algorithm) to find the execution order."""
        adj = {node_id: [] for node_id in self.nodes}
        in_degree = {node_id: 0 for node_id in self.nodes}

        # First, add bridge dependencies to the graph
        # Bridges create implicit connections between all bridged nodes
        for b in self.bridges:
            from_node = b['from']
            to_node = b['to']
            if from_node not in self.nodes or to_node not in self.nodes:
                continue
            # Add bridge edge: from_node -> to_node (from runs before to)
            if to_node not in adj[from_node]:
                adj[from_node].append(to_node)
                in_degree[to_node] += 1

        for conn in self.connections:
            source = conn['from']
            target = conn['to']
            
            # Skip orphaned connections (connections to non-existent nodes)
            if source not in self.nodes or target not in self.nodes:
                continue
            
            targets = [target]
            # If target is bridged, the source effectively feeds the whole bridge group
            if target in self.bridge_groups:
                targets = self.bridge_groups[target]
            
            for t in targets:
                # Avoid self-loops if user connected bridged nodes explicitly
                if t != source:
                    # Check if edge already exists to avoid double counting
                    if t not in adj[source]:
                        adj[source].append(t)
                        in_degree[t] += 1

        queue = deque([node_id for node_id in self.nodes if in_degree[node_id] == 0])
        sorted_order = []

        while queue:
            u = queue.popleft()
            sorted_order.append(u)

            for v in adj[u]:
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    queue.append(v)
        
        if len(sorted_order) != len(self.nodes):
            # Cycle detected. We continue by breaking the cycle to allow execution.
            # We iteratively pick the remaining node with the lowest in-degree (heuristic)
            # and treat it as satisfied to proceed.
            remaining_nodes = list(set(self.nodes.keys()) - set(sorted_order))
            
            while len(sorted_order) < len(self.nodes):
                # Re-calculate in-degrees for remaining nodes based only on other remaining nodes
                # actually, we just need to pick one to break the deadlock.
                candidates = [n for n in self.nodes if n not in sorted_order]
                if not candidates:
                    break
                
                # Pick the first one (arbitrary break)
                next_node = candidates[0]
                sorted_order.append(next_node)
                
                # Simulate processing this node to free up its neighbors
                for v in adj[next_node]:
                    in_degree[v] -= 1
                    if in_degree[v] == 0 and v not in sorted_order:
                        queue.append(v)
                
                # Process any newly freed nodes
                while queue:
                    u = queue.popleft()
                    if u not in sorted_order:
                        sorted_order.append(u)
                        for v in adj[u]:
                            in_degree[v] -= 1
                            if in_degree[v] == 0:
                                queue.append(v)

        return sorted_order

    def validate(self, module_manager) -> dict:
        """Validates the flow for potential issues before execution."""
        issues = []
        warnings = []
        
        # Get enabled modules
        enabled_modules = {m['id']: m for m in module_manager.get_all_modules() if m.get('enabled', False)}
        
        # Check 1: Nodes referencing disabled or non-existent modules
        for node_id, node in self.nodes.items():
            module_id = node.get('moduleId')
            if module_id not in enabled_modules:
                issues.append({
                    'type': 'disabled_module',
                    'node_id': node_id,
                    'node_name': node.get('name'),
                    'module_id': module_id,
                    'message': f"Node '{node.get('name')}' references disabled module '{module_id}'"
                })
        
        # Check 2: Orphaned connections (referencing non-existent nodes)
        node_ids = set(self.nodes.keys())
        for conn in self.connections:
            if conn['from'] not in node_ids:
                issues.append({
                    'type': 'orphaned_connection',
                    'from_id': conn['from'],
                    'to_id': conn['to'],
                    'message': f"Connection from non-existent node '{conn['from']}'"
                })
            if conn['to'] not in node_ids:
                issues.append({
                    'type': 'orphaned_connection',
                    'from_id': conn['from'],
                    'to_id': conn['to'],
                    'message': f"Connection to non-existent node '{conn['to']}'"
                })
        
        # Check 3: Tools enabled in System Prompt that don't exist or are disabled
        for node_id, node in self.nodes.items():
            if node.get('nodeTypeId') == 'system_prompt':
                enabled_tools = node.get('config', {}).get('enabled_tools', [])
                for tool_name in enabled_tools:
                    # Check if tool exists in tools.json
                    try:
                        from modules.tools.router import load_tools
                        tools = load_tools()
                        tool_config = tools.get(tool_name, {})
                        if tool_name not in tools:
                            warnings.append({
                                'type': 'missing_tool',
                                'node_id': node_id,
                                'node_name': node.get('name'),
                                'tool_name': tool_name,
                                'message': f"Node '{node.get('name')}' enables non-existent tool '{tool_name}'"
                            })
                        elif tool_config.get('enabled', True) == False:
                            warnings.append({
                                'type': 'disabled_tool',
                                'node_id': node_id,
                                'node_name': node.get('name'),
                                'tool_name': tool_name,
                                'message': f"Node '{node.get('name')}' enables disabled tool '{tool_name}'"
                            })
                    except Exception:
                        pass
        
        # Check 4: Check for nodes without any connections (might be unintentional)
        connected_nodes = set()
        for conn in self.connections:
            connected_nodes.add(conn['from'])
            connected_nodes.add(conn['to'])
        
        # Add bridged nodes to connected set
        for bridge in self.bridges:
            if bridge.get('from') in self.nodes:
                connected_nodes.add(bridge.get('from'))
            if bridge.get('to') in self.nodes:
                connected_nodes.add(bridge.get('to'))
        
        # Nodes that don't require connections (check various possible node type IDs)
        no_connection_required = [
            'trigger_node', 
            'repeater_node', 
            'annotation', 
            'comment', 
            'scheduled_start',
            'annotation_node',
            'comment_node'
        ]
        
        for node_id, node in self.nodes.items():
            node_type = node.get('nodeTypeId', '').lower()
            node_name = node.get('name', '').lower()
            
            # Check if it's an annotation/comment by name or type
            is_annotation = (node_type in no_connection_required or 
                           'annotation' in node_name or 
                           'comment' in node_name)
            
            if node_id not in connected_nodes and not is_annotation:
                warnings.append({
                    'type': 'unconnected_node',
                    'node_id': node_id,
                    'node_name': node.get('name'),
                    'message': f"Node '{node.get('name')}' has no connections"
                })
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings
        }

    async def run(self, initial_input: dict, start_node_id: str = None):
        """Executes the flow in order, passing data between nodes."""
        if settings.get("debug_mode"):
            debug_logger.log(self.flow_id, "SYSTEM", "FlowRunner", "flow_start", {"start_node": start_node_id, "input_source": initial_input.get("_input_source") if isinstance(initial_input, dict) else None})
        
        try:
            node_outputs = {}
            
            # Determine start nodes based on input source
            input_source = initial_input.get("_input_source") if isinstance(initial_input, dict) else None
            explicit_start_nodes = set()
            
            if start_node_id:
                # Explicit start node specified
                if start_node_id not in self.nodes:
                    raise ValueError(f"Start node {start_node_id} not found in flow.")
                execution_queue = deque([start_node_id])
                explicit_start_nodes.add(start_node_id)
            elif input_source:
                # Auto-detect which input node to start from based on source
                input_node_map = {
                    "chat": "chat_input",
                    "telegram": "telegram_input"
                }
                target_node_type = input_node_map.get(input_source)
                if target_node_type:
                    # Find all nodes of this type
                    start_nodes = [nid for nid, n in self.nodes.items() if n.get("nodeTypeId") == target_node_type]
                    if start_nodes:
                        execution_queue = deque(start_nodes)
                        explicit_start_nodes.update(start_nodes)
                    else:
                        # Fallback: find nodes with no incoming edges
                        execution_queue = deque([nid for nid in self.nodes if not any(c['to'] == nid for c in self.connections)])
                        explicit_start_nodes.update(execution_queue)
                else:
                    execution_queue = deque(self.execution_order)
            else:
                # No source specified, use topological order
                execution_queue = deque(self.execution_order)
            
            # Track how many times a node has run to prevent infinite loops
            node_run_counts = {node_id: 0 for node_id in self.nodes}
            max_loops = settings.get("max_node_loops", 1000)
            
            while execution_queue:
                node_id = execution_queue.popleft()
                
                if max_loops > 0 and node_run_counts[node_id] >= max_loops:
                    print(f"Warning: Node {node_id} hit max execution limit ({max_loops}). Stopping branch.")
                    continue
                
                node_run_counts[node_id] += 1
                
                node_meta = self.nodes[node_id]
                module_id = node_meta['moduleId']
                node_type_id = node_meta['nodeTypeId']
                
                if settings.get("debug_mode"):
                    debug_logger.log(self.flow_id, node_id, node_meta['name'], "start", {"input": node_input if 'node_input' in locals() else "Pending Input Resolution"})

                # 1. Determine Input Data (DAG Logic)
                incoming_edges = [c for c in self.connections if c['to'] == node_id]
                
                # Check if this node has upstream bridge dependencies that haven't run yet
                if node_id in self.bridge_groups and node_id not in explicit_start_nodes:
                    bridge_chain = self._get_bridge_order(node_id)
                    bridge_input = {k: v for k, v in initial_input.items() if k != "_input_source"} if isinstance(initial_input, dict) else initial_input
                    
                    for bridge_node_id in bridge_chain:
                        if bridge_node_id != node_id and bridge_node_id not in node_outputs:
                            bridge_meta = self.nodes[bridge_node_id]
                            bridge_module_id = bridge_meta['moduleId']
                            bridge_type_id = bridge_meta['nodeTypeId']
                            
                            bridge_cache_key = f"{bridge_module_id}.{bridge_type_id}"
                            if bridge_cache_key not in self._executor_cache:
                                node_dispatcher = importlib.import_module(f"modules.{bridge_module_id}.node")
                                if isinstance(node_dispatcher, types.ModuleType) and node_dispatcher.__name__ in sys.modules:
                                    importlib.reload(node_dispatcher)
                                bridge_executor_class = await node_dispatcher.get_executor_class(bridge_type_id)
                                self._executor_cache[bridge_cache_key] = bridge_executor_class
                            else:
                                bridge_executor_class = self._executor_cache[bridge_cache_key]
                            
                            if bridge_executor_class:
                                bridge_executor = bridge_executor_class()
                                bridge_config = (bridge_meta.get('config') or {}).copy()
                                bridge_config['_flow_id'] = self.flow_id
                                bridge_config['_node_id'] = bridge_node_id
                                
                                bridge_processed = await bridge_executor.receive(bridge_input, config=bridge_config)
                                if bridge_processed is None:
                                    break
                                bridge_output = await bridge_executor.send(bridge_processed)
                                node_outputs[bridge_node_id] = bridge_output
                                # Pass output as input to next bridge node
                                if isinstance(bridge_output, dict):
                                    bridge_input = bridge_input.copy()
                                    bridge_input.update(bridge_output)
                
                if node_id in explicit_start_nodes:
                    # Explicit start node receives the initial input directly
                    if isinstance(initial_input, dict):
                        node_input = {k: v for k, v in initial_input.items() if k != "_input_source"}
                    else:
                        node_input = initial_input
                elif not incoming_edges:
                    # Source node: receives global initial input (only if no explicit start nodes defined)
                    if explicit_start_nodes:
                        # Skip - we're in selective mode and this isn't a chosen start node
                        node_outputs[node_id] = None
                        continue
                    if isinstance(initial_input, dict):
                        node_input = {k: v for k, v in initial_input.items() if k != "_input_source"}
                    else:
                        node_input = initial_input
                else:
                    # Gather outputs from parents (including parents of bridged peers)
                    parent_outputs = []
                    
                    # Identify all relevant incoming edges
                    relevant_edges = list(incoming_edges)
                    if node_id in self.bridge_groups:
                        peers = self.bridge_groups[node_id]
                        # Include outputs from bridge peer nodes that have already run
                        for peer_id in peers:
                            if peer_id != node_id and peer_id in node_outputs:
                                peer_output = node_outputs.get(peer_id)
                                if peer_output is not None:
                                    parent_outputs.append(peer_output)
                            # Also get edges to peers
                            peer_edges = [c for c in self.connections if c['to'] == peer_id]
                            relevant_edges.extend(peer_edges)
                    
                    for edge in relevant_edges:
                        if edge['from'] in node_outputs:
                            p_out = node_outputs.get(edge['from'])
                            if p_out is not None:
                                parent_outputs.append(p_out)

                    if not parent_outputs:
                        # Branch stopped (condition failed upstream) or no data
                        node_outputs[node_id] = None
                        continue
                    
                    # Merge inputs from multiple parents
                    node_input = {}
                    for po in parent_outputs:
                        if isinstance(po, dict):
                            node_input.update(po)
                        else:
                            # Fallback for non-dict outputs
                            node_input = po

                if settings.get("debug_mode"):
                    debug_logger.log(self.flow_id, node_id, node_meta['name'], "input_resolved", {"input": node_input})

                try:
                    cache_key = f"{module_id}.{node_type_id}"
                    
                    if cache_key not in self._executor_cache:
                        # Dynamically import the module's node logic dispatcher
                        node_dispatcher = importlib.import_module(f"modules.{module_id}.node")
                        # Only reload if it's a real module (not a mock during testing)
                        if isinstance(node_dispatcher, types.ModuleType) and node_dispatcher.__name__ in sys.modules:
                            importlib.reload(node_dispatcher) # Ensure we have the latest code
                        # Get the specific executor class for this node type
                        executor_class = await node_dispatcher.get_executor_class(node_type_id)
                        self._executor_cache[cache_key] = executor_class

                    executor_class = self._executor_cache[cache_key]
                    if not executor_class:
                        # Fallback: Pass through if no executor found (e.g. missing module)
                        node_outputs[node_id] = node_input
                        continue

                    executor = executor_class()
                    node_config = (node_meta.get('config') or {}).copy()
                    node_config['_flow_id'] = self.flow_id
                    node_config['_node_id'] = node_id
                    
                    processed_data = await executor.receive(node_input, config=node_config)
                    
                    # If receive returns None (e.g. Condition failed), we stop this branch
                    if processed_data is None:
                        if settings.get("debug_mode"):
                            debug_logger.log(self.flow_id, node_id, node_meta['name'], "branch_stop", {"reason": "Node returned None"})
                        node_outputs[node_id] = None
                        continue

                    output = await executor.send(processed_data)
                    
                    # Automatic Context Propagation:
                    # If input had 'messages' (chat history) and output is a dict that missed it,
                    # preserve it. This ensures chains like System -> LLM -> Router -> Tools -> LLM
                    # maintain the conversation context.
                    if isinstance(node_input, dict) and "messages" in node_input:
                        if isinstance(output, dict) and "messages" not in output:
                            output["messages"] = node_input["messages"]

                    node_outputs[node_id] = output

                    if settings.get("debug_mode"):
                        debug_logger.log(self.flow_id, node_id, node_meta['name'], "end", {"output": output})
                    
                    # Routing Logic: Check if the node specified specific downstream targets
                    allowed_targets = None
                    if isinstance(output, dict) and "_route_targets" in output:
                        allowed_targets = output["_route_targets"]
                        if settings.get("debug_mode"):
                            debug_logger.log(self.flow_id, node_id, node_meta['name'], "routing", {"targets": allowed_targets})
                    
                    # If successful, add downstream nodes to queue if they aren't already pending
                    # This enables loops: A -> B -> A
                    downstream_nodes = [c['to'] for c in self.connections if c['from'] == node_id]
                    
                    # Note: Bridge handling is now done in _compute_execution_order via topological sort.
                    # No need to re-add bridge peers at runtime.

                    for child_id in downstream_nodes:
                        # If routing is active, check if child is allowed
                        if allowed_targets is not None:
                            # If the current node (parent) was the routing target, clear routing for its children
                            # This ensures routing only applies one level deep
                            parent_was_routed_to = node_id in allowed_targets
                            
                            if parent_was_routed_to:
                                # Clear routing so children can be processed normally
                                if isinstance(output, dict) and "_route_targets" in output:
                                    del output["_route_targets"]
                                allowed_targets = None
                            elif child_id not in allowed_targets:
                                if settings.get("debug_mode"):
                                    child_name = self.nodes.get(child_id, {}).get('name', child_id)
                                    debug_logger.log(self.flow_id, node_id, node_meta['name'], "routing_skip", {"skipped": child_name})
                                continue

                        if child_id not in execution_queue:
                            execution_queue.append(child_id)
                            if settings.get("debug_mode"):
                                child_name = self.nodes.get(child_id, {}).get('name', child_id)
                                debug_logger.log(self.flow_id, node_id, node_meta['name'], "queue_next", {"next": child_name})

                except (ImportError, AttributeError) as e:
                    print(f"Warning: Could not find or use node logic for {module_id}/{node_type_id}. Error: {e}. Passing data through.")
                    node_outputs[node_id] = node_input
                except Exception as e:
                    error_msg = f"Execution failed at node '{node_meta['name']}': {e}"
                    if settings.get("debug_mode"):
                        debug_logger.log(self.flow_id, node_id, node_meta['name'], "error", {"error": str(e)})
                    print(f"Error in FlowRunner: {error_msg}")
                    # Return a structured error that the chat UI can display
                    return {"error": error_msg}
            
            if settings.get("debug_mode"):
                debug_logger.log(self.flow_id, "SYSTEM", "FlowRunner", "flow_complete", {})
                
            # Return the output of the last executed node in the list that isn't None
            for node_id in reversed(self.execution_order):
                out = node_outputs.get(node_id)
                if out is not None:
                    return out
                    
            return {}
        except asyncio.CancelledError:
            if settings.get("debug_mode"):
                debug_logger.log(self.flow_id, "SYSTEM", "FlowRunner", "flow_cancelled", {})
            print(f"[FlowRunner] Flow {self.flow_id} cancelled")
            raise
        except Exception as e:
            if settings.get("debug_mode"):
                debug_logger.log(self.flow_id, "SYSTEM", "FlowRunner", "flow_error", {"error": str(e)})
            print(f"[FlowRunner] Flow {self.flow_id} failed: {e}")
            raise