import importlib
from collections import deque
from .flow_manager import flow_manager

class FlowRunner:
    _executor_cache = {}

    @classmethod
    def clear_cache(cls):
        cls._executor_cache.clear()

    def __init__(self, flow_id: str):
        self.flow = flow_manager.get_flow(flow_id)
        if not self.flow:
            raise ValueError(f"Flow with id {flow_id} not found.")
        
        self.nodes = {node['id']: node for node in self.flow['nodes']}
        self.connections = self.flow['connections']
        self.execution_order = self._topological_sort()

    def _topological_sort(self):
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
            raise Exception("Flow contains a cycle and cannot be executed.")

        return sorted_order

    async def run(self, initial_input: dict):
        """Executes the flow in order, passing data between nodes."""
        node_outputs = {}
        
        for node_id in self.execution_order:
            node_meta = self.nodes[node_id]
            module_id = node_meta['moduleId']
            node_type_id = node_meta['nodeTypeId']
            
            # 1. Determine Input Data (DAG Logic)
            incoming_edges = [c for c in self.connections if c['to'] == node_id]
            
            if not incoming_edges:
                # Source node: receives global initial input
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

            try:
                cache_key = f"{module_id}.{node_type_id}"
                
                if cache_key not in self._executor_cache:
                    # Dynamically import the module's node logic dispatcher
                    node_dispatcher = importlib.import_module(f"modules.{module_id}.node")
                    # Get the specific executor class for this node type
                    executor_class = await node_dispatcher.get_executor_class(node_type_id)
                    self._executor_cache[cache_key] = executor_class

                executor_class = self._executor_cache[cache_key]
                if not executor_class:
                    # Fallback: Pass through if no executor found (e.g. missing module)
                    node_outputs[node_id] = node_input
                    continue

                executor = executor_class()
                node_config = node_meta.get('config', {})
                
                processed_data = await executor.receive(node_input, config=node_config)
                
                # If receive returns None (e.g. Condition failed), we stop this branch
                if processed_data is None:
                    node_outputs[node_id] = None
                    continue

                output = await executor.send(processed_data)
                node_outputs[node_id] = output

            except (ImportError, AttributeError) as e:
                print(f"Warning: Could not find or use node logic for {module_id}/{node_type_id}. Error: {e}. Passing data through.")
                node_outputs[node_id] = node_input
            except Exception as e:
                error_msg = f"Execution failed at node '{node_meta['name']}': {e}"
                print(f"Error in FlowRunner: {error_msg}")
                # Return a structured error that the chat UI can display
                return {"error": error_msg}
            
        # Return the output of the last executed node in the list that isn't None
        for node_id in reversed(self.execution_order):
            out = node_outputs.get(node_id)
            if out is not None:
                return out
                
        return {}