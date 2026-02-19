import re

class ConditionExecutor:
    async def receive(self, input_data: dict, config: dict = None) -> dict:
        if input_data is None: return None
        
        config = config or {}
        target_field = config.get("target_field", "content")
        operator = config.get("operator", "contains")
        value = config.get("value", "")
        
        # Extract data to check
        data_to_check = input_data.get(target_field)
        
        # Special handling for 'messages' list to check the last user message
        if target_field == "messages" and isinstance(data_to_check, list):
             # Find last user message or just last message
             if data_to_check:
                 data_to_check = data_to_check[-1].get("content", "")
             else:
                 data_to_check = ""
        
        # Handle nested choices (LLM output) if checking content
        if target_field == "content" and not data_to_check and "choices" in input_data:
             try:
                 data_to_check = input_data["choices"][0]["message"]["content"]
             except:
                 pass

        if data_to_check is None:
            data_to_check = ""
            
        data_str = str(data_to_check)
        
        matched = False
        if operator == "contains":
            matched = value.lower() in data_str.lower()
        elif operator == "equals":
            matched = value.lower() == data_str.lower()
        elif operator == "not_equals":
            matched = value.lower() != data_str.lower()
        elif operator == "regex":
            try:
                matched = bool(re.search(value, data_str, re.IGNORECASE))
            except:
                matched = False
        elif operator == "exists":
            matched = bool(data_to_check)
            
        if matched:
            return input_data
        else:
            return None

    async def send(self, processed_data: dict) -> dict:
        return processed_data

async def get_executor_class(node_type_id: str):
    if node_type_id == "condition_node":
        return ConditionExecutor
    return None