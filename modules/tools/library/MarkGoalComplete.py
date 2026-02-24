import httpx

goal_id = args.get('goal_id')

try:
    response = httpx.post(
        f"http://localhost:8000/memory/goals/{goal_id}/complete",
        timeout=10.0
    )
    if response.status_code == 200:
        result = f"Goal {goal_id} marked as completed!"
    else:
        result = f"Failed to complete goal: {response.text}"
except Exception as e:
    result = f"Error completing goal: {str(e)}"
