class ChatInputExecutor:
    """
    Node to start a flow with chat data.
    """
    async def receive(self, input_data: dict, config: dict = None) -> dict:
        # This node's main job is to start the flow, so receive just accepts the data.
        return input_data

    async def send(self, processed_data: dict) -> dict:
        if "messages" not in processed_data:
            return {"error": "Flow started without 'messages'. 'Chat Input' node requires it."}
        # Pass the validated data onward.
        return processed_data

class ChatOutputExecutor:
    """
    Node to format the final AI response for the chat UI.
    """
    async def receive(self, input_data: dict, config: dict = None) -> dict:
        if "error" in input_data:
            return input_data
        
        # 1. Check for direct content (e.g. from a simple processor or memory recall acting as source)
        if "content" in input_data:
             return {"content": input_data["content"]}
             
        # 2. Check for OpenAI format (LLM)
        if "choices" in input_data:
            try:
                content = input_data.get("choices", [{}])[0].get("message", {}).get("content", "")
                if content:
                    return {"content": content}
            except (IndexError, AttributeError, TypeError):
                pass
        
        # 3. Fallback/Echo (if input was just messages)
        if "messages" in input_data and input_data["messages"]:
             return {"content": input_data["messages"][-1].get("content", "")}

        return {"content": "Flow finished but produced no valid response."}

    async def send(self, processed_data: dict) -> dict:
        # This is the final node, so it sends the simplified data back to the FlowRunner.
        return processed_data

async def get_executor_class(node_type_id: str):
    """Acts as a dispatcher to get the correct node executor class."""
    if node_type_id == 'chat_input':
        return ChatInputExecutor
    if node_type_id == 'chat_output':
        return ChatOutputExecutor
    return None