import re
import asyncio
import json
from core.debug import debug_logger

class DelayExecutor:
    async def receive(self, input_data: dict, config: dict = None) -> dict:
        if input_data is None: return None
        config = config or {}
        try:
            seconds = float(config.get("seconds", 1.0))
        except:
            seconds = 1.0
        
        if seconds < 0:
            seconds = 0
        await asyncio.sleep(seconds)
        return input_data

    async def send(self, processed_data: dict) -> dict:
        return processed_data

class ScriptExecutor:
    async def receive(self, input_data: dict, config: dict = None) -> dict:
        if input_data is None: return None
        config = config or {}
        code = config.get("code", "")
        
        if not code:
            return input_data
            
        # Define scope
        if isinstance(input_data, dict):
            data_val = input_data.copy()
            result_val = input_data.copy()
        else:
            # Handle non-dict inputs (lists, strings, etc.) gracefully
            data_val = input_data
            result_val = {}
            
        local_scope = {"data": data_val, "result": result_val, "json": json, "re": re}
        
        try:
            exec(code, local_scope)
            return local_scope.get("result")
        except Exception as e:
            print(f"Script Node Error: {e}")
            # Return error in data stream so it can be debugged
            if isinstance(input_data, dict):
                err_data = input_data.copy()
            else:
                err_data = {"original_input": input_data}
            err_data["error"] = f"Script failed: {str(e)}"
            return err_data

    async def send(self, processed_data: dict) -> dict:
        return processed_data

class RepeaterExecutor:
    async def receive(self, input_data: dict, config: dict = None) -> dict:
        if input_data is None: return None
        config = config or {}
        
        try:
            delay = float(config.get("delay", 5.0))
            max_repeats = int(config.get("max_repeats", 1))
        except (ValueError, TypeError):
            delay = 5.0
            max_repeats = 1
            
        flow_id = config.get("_flow_id")
        node_id = config.get("_node_id")
        current_repeat = input_data.get("_repeat_count", 0)
        
        # If max_repeats is 0, it loops forever. Otherwise, it checks the count.
        if flow_id and (max_repeats == 0 or current_repeat < max_repeats):
            async def trigger_next(fid, data, count, start_node):
                await asyncio.sleep(delay)
                try:
                    from core.flow_runner import FlowRunner
                    runner = FlowRunner(fid)
                    next_data = data.copy()
                    next_data["_repeat_count"] = count + 1
                    await runner.run(next_data, start_node_id=start_node)
                except Exception as e:
                    print(f"Repeater failed to trigger next run: {e}")
                    debug_logger.log(fid, "repeater_node", "Repeater", "error", f"Loop failed: {str(e)}")
            
            asyncio.create_task(trigger_next(flow_id, input_data, current_repeat, node_id))
            
        return input_data

    async def send(self, processed_data: dict) -> dict:
        return processed_data

class ConditionalRouterExecutor:
    async def receive(self, input_data: dict, config: dict = None) -> dict:
        """
        Routes data flow based on tool existence.
        Returns the input_data if tool exists, None otherwise.
        
        Config options:
        - check_field: field to check for existence (default: "tool_calls")
        """
        if input_data is None:
            return None
            
        config = config or {}
        check_field = config.get("check_field", "tool_calls")
        
        # Determine condition
        condition_met = False
        if isinstance(input_data, dict):
            if input_data.get(check_field):
                condition_met = True
            
            # Check OpenAI format for tool_calls if not found at top level
            elif check_field == "tool_calls":
                try:
                    choices = input_data.get("choices")
                    if isinstance(choices, list) and len(choices) > 0:
                        message = choices[0].get("message")
                        if isinstance(message, dict) and message.get("tool_calls"):
                            condition_met = True
                except Exception:
                    pass
        
        if config.get("invert", False):
            condition_met = not condition_met
            
        # Determine targets based on config
        targets = config.get("true_branches", []) if condition_met else config.get("false_branches", [])
            
        # Return data with routing info
        result = input_data.copy() if isinstance(input_data, dict) else {"content": input_data}
        result["_route_targets"] = targets
        return result

    async def send(self, processed_data: dict) -> dict:
        return processed_data

class ReasoningLoadExecutor:
    async def receive(self, input_data: dict, config: dict = None) -> dict:
        if input_data is None: return None
        config = config or {}
        last_n = int(config.get("last_n", 5))
        
        if isinstance(input_data, dict):
            history = input_data.get("reasoning_history", [])
            if isinstance(history, list) and len(history) > last_n:
                result = input_data.copy()
                result["reasoning_history"] = history[-last_n:]
                return result
        return input_data

    async def send(self, processed_data: dict) -> dict:
        return processed_data

class TriggerExecutor:
    async def receive(self, input_data: dict, config: dict = None) -> dict:
        # Pass-through: returns input data exactly as is
        return input_data

    async def send(self, processed_data: dict) -> dict:
        return processed_data

async def get_executor_class(node_type_id: str):
    if node_type_id == "delay_node":
        return DelayExecutor
    if node_type_id == "script_node":
        return ScriptExecutor
    if node_type_id == "repeater_node":
        return RepeaterExecutor
    if node_type_id == "conditional_router":
        return ConditionalRouterExecutor
    if node_type_id == "trigger_node":
        return TriggerExecutor
    if node_type_id == "reasoning_load":
        return ReasoningLoadExecutor
    return None