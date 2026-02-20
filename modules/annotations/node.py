class CommentExecutor:
    """
    Executor for the Comment Node.
    Since comments are purely visual annotations on the frontend and cannot be connected
    to other nodes, this executor is largely a placeholder to prevent backend errors
    during flow execution. It simply passes data through.
    """
    async def receive(self, input_data: dict, config: dict = None) -> dict:
        return input_data

    async def send(self, processed_data: dict) -> dict:
        return processed_data

async def get_executor_class(node_type_id: str):
    if node_type_id == "comment_node":
        return CommentExecutor
    return None