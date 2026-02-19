class SystemPromptExecutor:
    async def receive(self, input_data: dict, config: dict = None) -> dict:
        """
        Receives the current conversation state and prepends a system message.
        """
        config = config or {}
        # Default prompt if none is configured
        prompt_text = config.get("system_prompt", "You are NeuroCore, a helpful and intelligent AI assistant.")
        
        # Get existing messages from the flow data
        messages = input_data.get("messages", [])
        
        # Create the system message
        system_message = {"role": "system", "content": prompt_text}
        
        # Prepend the system message to the history
        new_messages = [system_message] + messages
        
        # Return the updated data structure
        # We use **input_data to preserve any other keys flowing through the system
        return {**input_data, "messages": new_messages}

    async def send(self, processed_data: dict) -> dict:
        return processed_data

async def get_executor_class(node_type_id: str):
    if node_type_id == "system_prompt":
        return SystemPromptExecutor
    return None