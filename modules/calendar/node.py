class SendReminderExecutor:
    async def receive(self, input_data: dict, config: dict = None) -> dict:
        """
        Receives data and formats it as a reminder notification.
        Useful for explicitly notifying the user about an event in the flow.
        """
        content = ""
        if isinstance(input_data, dict):
            content = input_data.get("content", "")
        elif isinstance(input_data, str):
            content = input_data
        
        # Format as a system notification
        notification = f"🔔 **REMINDER:** {content}"
        
        return {"content": notification}

    async def send(self, processed_data: dict) -> dict:
        return processed_data

async def get_executor_class(node_type_id: str):
    if node_type_id == 'send_reminder':
        return SendReminderExecutor
    return None