import httpx

goal_id = args.get('goal_id')

try:
    response = httpx.delete(
        f"http://localhost:8000/memory/goals/{goal_id}",
        timeout=10.0
    )
    if response.status_code == 200:
        result = f"Goal {goal_id} deleted!"
    else:
        result = f"Failed to delete goal: {response.text}"
except Exception as e:
    result = f"Error deleting goal: {str(e)}"
