from fastapi import APIRouter, Request, Query
from fastapi.responses import HTMLResponse, Response
from fastapi.templating import Jinja2Templates
import json
from modules.memory.backend import memory_store
from datetime import datetime
import asyncio
from functools import partial

router = APIRouter()
templates = Jinja2Templates(directory="web/templates")

# Helper filter for templates
def format_timestamp(ts):
    return datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M')

templates.env.filters["datetime"] = format_timestamp

@router.get("/gui", response_class=HTMLResponse)
async def browser_gui(request: Request):
    # Initial load with default browse parameters
    loop = asyncio.get_running_loop()
    memories = await loop.run_in_executor(memory_store.executor, partial(memory_store.browse, limit=50))
    return templates.TemplateResponse(request, "memory_browser.html", {
        "memories": memories
    })

@router.get("/list", response_class=HTMLResponse)
async def list_memories(
    request: Request, 
    q: str = Query(None),
    filter_date: str = Query("ALL")
):
    try:
        loop = asyncio.get_running_loop()
        memories = await loop.run_in_executor(memory_store.executor, partial(memory_store.browse, search_text=q, filter_date=filter_date, limit=50))
        return templates.TemplateResponse(request, "memory_list.html", {"memories": memories})
    except Exception as e:
        return HTMLResponse(f"<div class='p-4 text-red-400 italic'>Error loading memories: {str(e)}</div>")

@router.delete("/delete/{memory_id}")
async def delete_memory(request: Request, memory_id: int):
    try:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(memory_store.executor, partial(memory_store.delete_entry, memory_id))
        return Response(status_code=200, headers={"HX-Trigger": json.dumps({"showMessage": {"level": "info", "message": "Memory deleted"}})})
    except Exception as e:
        return Response(status_code=200, headers={"HX-Trigger": json.dumps({"showMessage": {"level": "error", "message": f"Delete failed: {str(e)}"}})})