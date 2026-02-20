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

    def __init__(self, flow_id: str):
        self.flow_id = flow_id
        self.flow = flow_manager.get_flow(flow_id)
        if not self.flow:
            raise ValueError(f"Flow with id {flow_id} not found.")
        
        self.nodes = {node['id']: node for node in self.flow['nodes']}
        self.connections = self.flow['connections']
        self.execution_order = self._compute_execution_order()

    def _compute_execution_order(self):
        """Performs a topological sort (Kahn's algorithm) to find the execution order."""
        adj = {node_id: [] for node_id in self.nodes}
        in_degree = {node_id: 0 for node_id in self.nodes}

        for conn in self.connections:
            adj[conn['from']].append(conn['to'])
            in_degree[conn['to']] += 1

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

    async def run(self, initial_input: dict):
        """Executes the flow in order, passing data between nodes."""
        node_outputs = {}
        
        # Use a queue for execution to support cycles (re-execution of nodes)
        execution_queue = deque(self.execution_order)
        
        # Track how many times a node has run to prevent infinite loops
        node_run_counts = {node_id: 0 for node_id in self.nodes}
        MAX_NODE_RUNS = 20 
        
        while execution_queue:
            node_id = execution_queue.popleft()
            
            if node_run_counts[node_id] >= MAX_NODE_RUNS:
                print(f"Warning: Node {node_id} hit max execution limit ({MAX_NODE_RUNS}). Stopping branch.")
                continue
            
            node_run_counts[node_id] += 1
            
            node_meta = self.nodes[node_id]
            module_id = node_meta['moduleId']
            node_type_id = node_meta['nodeTypeId']
            
            if settings.get("debug_mode"):
                debug_logger.log(self.flow_id, node_id, node_meta['name'], "start", {"input": node_input if 'node_input' in locals() else "Pending Input Resolution"})

            # 1. Determine Input Data (DAG Logic)
            incoming_edges = [c for c in self.connections if c['to'] == node_id]
            
            if not incoming_edges:
                # Source node: receives global initial input
                if isinstance(initial_input, dict):
                    node_input = initial_input.copy()
                else:
                    node_input = initial_input
            else:
                # Gather outputs from parents
                parent_outputs = []
                for edge in incoming_edges:
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
                
                processed_data = await executor.receive(node_input, config=node_config)
                
                # If receive returns None (e.g. Condition failed), we stop this branch
                if processed_data is None:
                    node_outputs[node_id] = None
                    continue

                output = await executor.send(processed_data)
                node_outputs[node_id] = output

                if settings.get("debug_mode"):
                    debug_logger.log(self.flow_id, node_id, node_meta['name'], "end", {"output": output})
                
                # Routing Logic: Check if the node specified specific downstream targets
                allowed_targets = None
                if isinstance(output, dict) and "_route_targets" in output:
                    allowed_targets = output["_route_targets"]
                
                # If successful, add downstream nodes to queue if they aren't already pending
                # This enables loops: A -> B -> A
                downstream_nodes = [c['to'] for c in self.connections if c['from'] == node_id]
                for child_id in downstream_nodes:
                    # If routing is active, only queue allowed targets
                    if allowed_targets is not None and child_id not in allowed_targets:
                        continue

                    if child_id not in execution_queue:
                        execution_queue.append(child_id)

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
            
        # Return the output of the last executed node in the list that isn't None
        for node_id in reversed(self.execution_order):
            out = node_outputs.get(node_id)
            if out is not None:
                return out
                
        return {}