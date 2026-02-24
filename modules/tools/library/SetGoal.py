import httpx
import json

description = args.get('description')
priority = args.get('priority', 0)
deadline = args.get('deadline')

try:
    response = httpx.post(
        "http://localhost:8000/memory/goals",
        data={"description": description, "priority": priority, "deadline": deadline or ""},
        timeout=10.0
    )
    if response.status_code == 200:
        data = response.json()
        result = f"Goal created successfully! Goal ID: {data.get('goal_id')}"
    else:
        result = f"Failed to create goal: {response.text}"
except Exception as e:
    result = f"Error creating goal: {str(e)}"
