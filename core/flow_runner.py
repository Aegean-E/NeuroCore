import asyncio
import importlib
import json
import logging
import sys
import types
from collections import deque, defaultdict
from .flow_manager import flow_manager
from core.settings import settings
from core.debug import debug_logger
from core.errors import FlowError, NodeExecutionError

logger = logging.getLogger(__name__)

class FlowRunner:
    _executor_cache = {}
    _max_cache_size = 100  # Maximum number of cached executor classes
    _cache_lock = None  # Lazy-initialized lock for thread-safe cache operations
    
    @classmethod
    def clear_cache(cls):
        cls._executor_cache.clear()
    
    @classmethod
    def _get_cache_lock(cls):
        """Lazy-initialize the lock to avoid attaching to wrong event loop."""
        if cls._cache_lock is None:
            cls._cache_lock = asyncio.Lock()
        return cls._cache_lock
    
    @classmethod
    async def _get_executor_class(cls, module_id: str, node_type_id: str):
        """Thread-safe method to get or create an executor class from the cache."""
        cache_key = f"{module_id}.{node_type_id}"
        
        async with cls._get_cache_lock():
            # Check if already in cache (after acquiring lock)
            if cache_key in cls._executor_cache:
                return cls._executor_cache[cache_key]
            
            # Manage cache size before adding new entry
            cls._manage_cache_size()
            
            # Dynamically import the module's node logic dispatcher
            node_dispatcher = importlib.import_module(f"modules.{module_id}.node")
            # Only reload in debug mode to pick up hot-code changes;
            # in production this is wasteful and can cause state issues.
            if settings.get("debug_mode") and isinstance(node_dispatcher, types.ModuleType) and node_dispatcher.__name__ in sys.modules:
                # Invalidate cache before reloading to ensure fresh class is used
                cls._executor_cache.pop(cache_key, None)
                importlib.reload(node_dispatcher)
            # Get the specific executor class for this node type
            executor_class = await node_dispatcher.get_executor_class(node_type_id)
            cls._executor_cache[cache_key] = executor_class
            return executor_class
    
    @classmethod
    def _manage_cache_size(cls):
        """Ensure cache doesn't grow indefinitely by removing oldest entries when limit reached."""
        if len(cls._executor_cache) >= cls._max_cache_size:
            # Remove oldest 20% of entries (FIFO eviction)
            # Cap at len(cache) // 5 to avoid issues when cache is small
            num_to_remove = max(1, len(cls._executor_cache) // 5)
            keys_to_remove = list(cls._executor_cache.keys())[:num_to_remove]
            for key in keys_to_remove:
                del cls._executor_cache[key]
            if settings.get("debug_mode"):
                logger.debug(f"[FlowRunner] Cache size limit reached. Removed {len(keys_to_remove)} oldest entries.")

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
        
        # Precompute incoming edges adjacency dict for O(1) lookups
        self.incoming_edges = defaultdict(list)
        for c in self.connections:
            self.incoming_edges[c['to']].append(c)
        
        # Precompute downstream nodes for O(1) lookups
        self.downstream_nodes = defaultdict(list)
        for c in self.connections:
            self.downstream_nodes[c['from']].append(c['to'])
        
        self.execution_order = self._compute_execution_order()
        
        if settings.get("debug_mode"):
            logger.debug(f"[FlowRunner] Initialized for flow {flow_id}")

    def _build_bridge_groups(self):
        """
        Identifies groups of bridged nodes using BFS to find connected components.
        
        Bridges are bidirectional connections that create implicit execution dependencies.
        When nodes A-B-C are bridged together, they form a "bridge group" where:
        - All nodes in the group share the same input data
        - The group executes in upstream-to-downstream order
        - Outputs from earlier nodes in the chain are merged into later nodes
        
        Algorithm:
        1. Build adjacency list from bridge connections (undirected graph)
        2. Use BFS to find all connected components
        3. Map each node to its full component group
        
        Returns:
            dict: Mapping of node_id -> list of all node_ids in the same bridge group
        """
        # Build undirected adjacency list: bridges connect nodes bidirectionally
        adj = {node_id: [] for node_id in self.nodes}
        for b in self.bridges:
            if b['from'] in adj and b['to'] in adj:
                adj[b['from']].append(b['to'])
                adj[b['to']].append(b['from'])
        
        groups = {}
        visited = set()
        for node_id in self.nodes:
            if node_id in visited: 
                continue
            if not adj[node_id]: 
                continue
            
            # BFS to find all nodes in this connected component
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
            
            # Each node in the component maps to the full component list
            for member in component:
                groups[member] = component
        return groups

    
    def _get_bridge_order(self, node_id):
        """
        Get bridge chain nodes in execution order (upstream to downstream).
        
        Bridges have a direction: from_node executes BEFORE to_node. This method
        performs a topological sort within the bridge group to determine the
        correct execution order.
        
        For example, if bridges are A->B and B->C, the order is [A, B, C].
        This ensures that when node C executes, it has access to outputs from A and B.
        
        Args:
            node_id: The ID of a node in the bridge group
            
        Returns:
            list: Ordered list of node IDs from furthest upstream to downstream
        """
        if node_id not in self.bridge_groups:
            return []
        
        # Get all nodes in this bridge group
        group = self.bridge_groups[node_id]
        ordered = []
        visited = set()
        
        # DFS to build topological order (visit dependencies first)
        def visit(nid):
            if nid in visited:
                return
            visited.add(nid)
            # Visit all upstream nodes first (nodes that have edges TO this node)
            for b in self.bridges:
                if b['to'] == nid and b['from'] in group:
                    visit(b['from'])
            ordered.append(nid)
        
        # Visit all nodes in the group to build complete ordering
        for nid in group:
            visit(nid)
        
        return ordered


    def _compute_execution_order(self):
        """
        Performs a topological sort (Kahn's algorithm) to find the execution order.
        
        This method builds a directed acyclic graph (DAG) from:
        1. Bridge connections (implicit dependencies between bridged nodes)
        2. Regular connections (explicit user-defined edges)
        
        Bridge handling: When a node is bridged, all nodes in its bridge group
        receive the same input. The bridge edges define execution order within
        the group, while connections to any group member feed the entire group.
        
        Cycle handling: If cycles are detected (e.g., A->B->A), we break them
        arbitrarily by picking remaining nodes and forcing them into the order.
        This allows execution to continue even with malformed flows.
        
        Returns:
            list: Topologically sorted list of node IDs for execution
        """
        # Build adjacency list and in-degree count for Kahn's algorithm
        adj = {node_id: [] for node_id in self.nodes}
        in_degree = {node_id: 0 for node_id in self.nodes}

        # Step 1: Add bridge dependencies to the graph
        # Bridges are directed: from_node must execute before to_node
        for b in self.bridges:
            from_node = b['from']
            to_node = b['to']
            if from_node not in self.nodes or to_node not in self.nodes:
                continue
            # Add bridge edge: from_node -> to_node
            if to_node not in adj[from_node]:
                adj[from_node].append(to_node)
                in_degree[to_node] += 1

        # Step 2: Add regular connections to the graph
        for conn in self.connections:
            source = conn['from']
            target = conn['to']
            
            # Skip orphaned connections (connections to non-existent nodes)
            if source not in self.nodes or target not in self.nodes:
                continue
            
            # If target is bridged, the source feeds the entire bridge group
            targets = [target]
            if target in self.bridge_groups:
                targets = self.bridge_groups[target]
            
            for t in targets:
                # Avoid self-loops if user connected bridged nodes explicitly
                if t != source:
                    # Check if edge already exists to avoid double counting
                    if t not in adj[source]:
                        adj[source].append(t)
                        in_degree[t] += 1

        # Step 3: Kahn's algorithm - start with nodes that have no dependencies
        queue = deque([node_id for node_id in self.nodes if in_degree[node_id] == 0])
        sorted_order = []

        while queue:
            u = queue.popleft()
            sorted_order.append(u)

            # Reduce in-degree for all neighbors
            for v in adj[u]:
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    queue.append(v)
        
        # Step 4: Handle cycles - if not all nodes were sorted, we have a cycle
        if len(sorted_order) != len(self.nodes):
            # Cycle detected. Break it by arbitrarily adding remaining nodes.
            # This is a heuristic to allow execution even with malformed flows.
            remaining_nodes = list(set(self.nodes.keys()) - set(sorted_order))
            
            # Log warning about cycle detection - always log in production
            logger.warning(f"[FlowRunner] Cycle detected in flow. Breaking by adding remaining nodes: {remaining_nodes}")
            
            # Check for intentional loops (nodes with isReverted flag indicating loop nodes)
            loop_nodes = [nid for nid, node in self.nodes.items() if node.get('isReverted', False)]
            if loop_nodes:
                logger.warning(f"[FlowRunner] Note: Flow contains {len(loop_nodes)} node(s) marked as loop (isReverted): {loop_nodes}")
            
            while len(sorted_order) < len(self.nodes):
                # Pick the first remaining node to break the deadlock
                candidates = [n for n in self.nodes if n not in sorted_order]
                if not candidates:
                    break
                
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
        # Load tools once outside the loop to avoid O(nodes × tools) disk reads
        tools = {}
        try:
            from modules.tools.router import load_tools
            tools = load_tools()
        except Exception as e:
            warnings.append({
                'type': 'tool_load_error',
                'node_id': None,
                'node_name': None,
                'tool_name': None,
                'message': f"Failed to load tools: {str(e)}"
            })
        
        for node_id, node in self.nodes.items():
            if node.get('nodeTypeId') == 'system_prompt':
                enabled_tools = node.get('config', {}).get('enabled_tools', [])
                for tool_name in enabled_tools:
                    # Check if tool exists in tools.json (using cached tools)
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

    async def run(self, initial_input: dict, start_node_id: str = None, timeout: float = None, raise_errors: bool = False, episode_id: str = None):
        """
        Executes the flow in order, passing data between nodes.
        
        Args:
            initial_input: Input data for the flow
            start_node_id: Optional specific node to start from
            timeout: Optional timeout in seconds for the entire flow execution
            raise_errors: If True, raise exceptions instead of returning error dicts
            episode_id: Optional episode ID for episode persistence support
        """
        # --- Episode Persistence Support ---
        sm = None
        episode = None
        enable_episode = episode_id is not None
        
        if enable_episode:
            try:
                from core.session_manager import get_session_manager, EpisodeState
                sm = get_session_manager()
                # Try to load the episode
                episode = sm.load_episode_by_id(episode_id)
                if episode:
                    # Restore state from episode to input
                    if episode.plan:
                        initial_input.setdefault("plan", episode.plan)
                    if episode.current_step is not None:
                        initial_input.setdefault("current_step", episode.current_step)
                    if episode.completed_steps:
                        initial_input.setdefault("completed_steps", episode.completed_steps)
                    # Update phase to executing
                    sm.save_episode_by_id(episode_id, phase=EpisodeState.PHASE_EXECUTING)
            except Exception:
                pass  # Don't fail if episode loading fails
        if settings.get("debug_mode"):
            debug_logger.log(self.flow_id, "SYSTEM", "FlowRunner", "flow_start", {"start_node": start_node_id, "input_source": initial_input.get("_input_source") if isinstance(initial_input, dict) else None, "timeout": timeout})
        
        async def run_impl():
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
            
            # Parallel set for O(1) membership checks instead of O(n) deque scan
            pending_nodes = set(execution_queue)
            
            while execution_queue:
                node_id = execution_queue.popleft()
                pending_nodes.discard(node_id)
                
                if max_loops > 0 and node_run_counts[node_id] >= max_loops:
                    logger.warning(f"Warning: Node {node_id} hit max execution limit ({max_loops}). Stopping branch.")
                    continue
                
                node_run_counts[node_id] += 1
                
                node_meta = self.nodes[node_id]
                module_id = node_meta['moduleId']
                node_type_id = node_meta['nodeTypeId']

                # 1. Determine Input Data (DAG Logic) - use precomputed incoming_edges for O(1) lookup
                incoming_edges = self.incoming_edges.get(node_id, [])
                
                # Bridge Execution: Process upstream bridge nodes before this node
                # Bridges create implicit execution chains where earlier nodes in the
                # chain feed their outputs to later nodes. This allows for patterns like:
                # Memory Recall -> System Prompt -> LLM Core (all bridged together)
                if node_id in self.bridge_groups and node_id not in explicit_start_nodes:
                    bridge_chain = self._get_bridge_order(node_id)
                    # Start with initial input (excluding the internal routing marker)
                    bridge_input = {k: v for k, v in initial_input.items() if k != "_input_source"} if isinstance(initial_input, dict) else initial_input
                    
                    # Execute each upstream bridge node in order
                    for bridge_node_id in bridge_chain:
                        if bridge_node_id != node_id and bridge_node_id not in node_outputs:
                            bridge_meta = self.nodes[bridge_node_id]
                            bridge_module_id = bridge_meta['moduleId']
                            bridge_type_id = bridge_meta['nodeTypeId']
                            
                            # Use thread-safe method to get executor class
                            bridge_executor_class = await self._get_executor_class(bridge_module_id, bridge_type_id)
                            
                            if bridge_executor_class:
                                try:
                                    bridge_executor = bridge_executor_class()
                                    bridge_config = (bridge_meta.get('config') or {}).copy()
                                    bridge_config['_flow_id'] = self.flow_id
                                    bridge_config['_node_id'] = bridge_node_id
                                    
                                    # Execute the bridge node's receive/send cycle
                                    bridge_processed = await bridge_executor.receive(bridge_input, config=bridge_config)
                                    if bridge_processed is None:
                                        # Node returned None - stop this branch
                                        break
                                    bridge_output = await bridge_executor.send(bridge_processed)
                                    node_outputs[bridge_node_id] = bridge_output
                                    
                                    # If send() returns None, stop the bridge chain to avoid stale input
                                    if bridge_output is None:
                                        break
                                    
                                    # Merge bridge output into input for next node in chain
                                    # This allows bridge nodes to progressively build context
                                    if isinstance(bridge_output, dict):
                                        bridge_input = bridge_input.copy()
                                        bridge_input.update(bridge_output)
                                except (ImportError, AttributeError, RuntimeError) as bridge_err:
                                    # Specific exception types for better error categorization:
                                    # - ImportError: Module not found or import failed
                                    # - AttributeError: Executor class or method missing
                                    # - RuntimeError: Node execution failed at runtime
                                    import traceback
                                    logger.error(f"[Bridge Error] Node {bridge_node_id} failed: {bridge_err}")
                                    logger.error(f"[Bridge Error] Traceback: {traceback.format_exc()}")
                                    if settings.get("debug_mode"):
                                        debug_logger.log(self.flow_id, bridge_node_id, bridge_meta.get('name', bridge_node_id), "bridge_error", {"error": str(bridge_err), "traceback": traceback.format_exc()})
                                    # Continue without bridge output - don't fail the whole flow
                                    break

                
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
                    bridge_outputs = []
                    
                    # Identify all relevant incoming edges
                    relevant_edges = list(incoming_edges)
                    if node_id in self.bridge_groups:
                        peers = self.bridge_groups[node_id]
                        # Include outputs from bridge peer nodes that have already run
                        for peer_id in peers:
                            if peer_id != node_id and peer_id in node_outputs:
                                peer_output = node_outputs.get(peer_id)
                                if peer_output is not None:
                                    bridge_outputs.append(peer_output)
                            # Also get edges to peers
                            peer_edges = [c for c in self.connections if c['to'] == peer_id]
                            relevant_edges.extend(peer_edges)
                    
                    for edge in relevant_edges:
                        if edge['from'] in node_outputs:
                            p_out = node_outputs.get(edge['from'])
                            if p_out is not None:
                                parent_outputs.append(p_out)
                    
                    # Deduplicate: Remove parent outputs that are also in bridge_outputs
                    # (This happens when a node is both a bridge peer and has a regular connection)
                    if bridge_outputs:
                        bridge_keys = set()
                        for bo in bridge_outputs:
                            if isinstance(bo, dict):
                                bridge_keys.add(id(bo))
                        parent_outputs = [po for po in parent_outputs if id(po) not in bridge_keys]

                    if not parent_outputs and not bridge_outputs:
                        # Branch stopped (condition failed upstream) or no data
                        node_outputs[node_id] = None
                        continue
                    
                    # Merge inputs: parent outputs first, then bridge outputs (bridge injects context)
                    # IMPORTANT: Merge in REVERSE order so the last bridge (closest to target) wins
                    node_input = {}
                    for po in parent_outputs:
                        if isinstance(po, dict):
                            node_input.update(po)
                    
                    # Merge bridge outputs in REVERSE order so the last bridge (closest to target) wins
                    for po in reversed(bridge_outputs):
                        if isinstance(po, dict):
                            node_input.update(po)  # Bridge outputs take precedence
                    
                    if settings.get("debug_mode") and bridge_outputs:
                        import json
                        try:
                            bridge_msg_preview = json.dumps(bridge_outputs[0].get("messages", [])[:2])
                        except (json.JSONDecodeError, TypeError, KeyError):
                            # json.JSONDecodeError: Failed to serialize messages
                            # TypeError: messages is not JSON serializable
                            # KeyError: messages key missing
                            bridge_msg_preview = str(bridge_outputs[0].get("messages", "N/A"))[:200]
                        debug_logger.log(self.flow_id, node_id, node_meta['name'], "bridge_merge", {
                            "bridge_count": len(bridge_outputs),
                            "bridge_msg_preview": bridge_msg_preview
                        })

                if settings.get("debug_mode"):
                    debug_logger.log(self.flow_id, node_id, node_meta['name'], "input_resolved", {"input": node_input})

                try:
                    # Use thread-safe method to get executor class
                    executor_class = await self._get_executor_class(module_id, node_type_id)
                    
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
                    # Nodes can opt-out by setting _strip_messages: true in their output
                    if isinstance(node_input, dict) and "messages" in node_input:
                        if isinstance(output, dict) and "messages" not in output:
                            # Check if node intentionally stripped messages
                            if not output.get("_strip_messages", False):
                                output["messages"] = node_input["messages"]
                            # Remove the marker if present
                            output.pop("_strip_messages", None)

                    node_outputs[node_id] = output

                    if settings.get("debug_mode"):
                        debug_logger.log(self.flow_id, node_id, node_meta['name'], "end", {"output": output})
                    
                    # Routing Logic: Check if the node specified specific downstream targets
                    # Consume _route_targets at the point it's read - this ensures routing
                    # only applies to children of the node that set the routing, then clears
                    # so grandchildren run freely
                    allowed_targets = None
                    if isinstance(output, dict) and "_route_targets" in output:
                        allowed_targets = output.pop("_route_targets")  # Consume it
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
                            # Check if this child is in the allowed targets from the router
                            if child_id not in allowed_targets:
                                if settings.get("debug_mode"):
                                    child_name = self.nodes.get(child_id, {}).get('name', child_id)
                                    debug_logger.log(self.flow_id, node_id, node_meta['name'], "routing_skip", {"skipped": child_name})
                                continue

                        if child_id not in pending_nodes:
                            execution_queue.append(child_id)
                            pending_nodes.add(child_id)
                            if settings.get("debug_mode"):
                                child_name = self.nodes.get(child_id, {}).get('name', child_id)
                                debug_logger.log(self.flow_id, node_id, node_meta['name'], "queue_next", {"next": child_name})

                except (ImportError, AttributeError) as e:
                    # Module import issues or missing executor classes
                    logger.warning(f"Warning: Could not find or use node logic for {module_id}/{node_type_id}. Error: {e}. Passing data through.")
                    node_outputs[node_id] = node_input
                except (RuntimeError, ValueError, TypeError) as e:
                    # Node execution errors: runtime failures, invalid values, type mismatches
                    error_msg = f"Execution failed at node '{node_meta['name']}': {e}"
                    if settings.get("debug_mode"):
                        debug_logger.log(self.flow_id, node_id, node_meta['name'], "error", {"error": str(e), "error_type": type(e).__name__})
                    logger.error(f"Error in FlowRunner: {error_msg}")
                    # Return a structured error that the chat UI can display
                    return {"error": error_msg}

            
            if settings.get("debug_mode"):
                debug_logger.log(self.flow_id, "SYSTEM", "FlowRunner", "flow_complete", {})
                
            # Return the output of the last executed node in the list that isn't None
            for node_id in reversed(self.execution_order):
                out = node_outputs.get(node_id)
                if out is not None:
                    # Save episode state if tracking
                    if sm and episode_id:
                        try:
                            from core.session_manager import EpisodeState
                            # Determine final phase based on result
                            final_phase = EpisodeState.PHASE_COMPLETED
                            if isinstance(out, dict) and out.get("error"):
                                final_phase = EpisodeState.PHASE_FAILED
                            
                            sm.save_episode_by_id(
                                episode_id,
                                phase=final_phase,
                                plan=out.get("plan", []),
                                current_step=out.get("current_step", 0),
                                completed_steps=out.get("completed_steps", []),
                            )
                        except Exception:
                            pass
                    return out
            
            # Save episode state if tracking (no successful output)
            if sm and episode_id:
                try:
                    from core.session_manager import EpisodeState
                    sm.save_episode_by_id(
                        episode_id,
                        phase=EpisodeState.PHASE_COMPLETED,
                    )
                except Exception:
                    pass
            
            return {}
        
        # Execute with optional timeout
        if timeout is not None and timeout > 0:
            try:
                return await asyncio.wait_for(run_impl(), timeout=timeout)
            except asyncio.TimeoutError:
                error_msg = f"Flow execution timed out after {timeout} seconds"
                if settings.get("debug_mode"):
                    debug_logger.log(self.flow_id, "SYSTEM", "FlowRunner", "flow_timeout", {"timeout": timeout})
                logger.error(f"[FlowRunner] {error_msg}")
                return {"error": error_msg}
        else:
            return await run_impl()
