from fastapi import APIRouter, Request, Form, Depends, Query
from fastapi.responses import HTMLResponse, Response
from core.settings import settings
from core.dependencies import get_llm_bridge
from modules.chat.sessions import session_manager
from fastapi.templating import Jinja2Templates
from core.flow_runner import FlowRunner
from core.flow_manager import flow_manager

router = APIRouter()
templates = Jinja2Templates(directory="web/templates")

@router.get("", response_class=HTMLResponse)
async def chat_page(request: Request):
    # Access module manager from app state to render the sidebar correctly
    module_manager = request.app.state.module_manager
    enabled_modules = [m for m in module_manager.get_all_modules() if m.get("enabled")]
    
    return templates.TemplateResponse(request, "index.html", {
        "modules": enabled_modules,
        "active_module": "chat",
        "sidebar_template": "chat_sidebar.html",
        "sessions": session_manager.list_sessions()
    })

@router.get("", response_class=HTMLResponse)
async def chat_page(request: Request):
    # Access module manager from app state to render the sidebar correctly
    module_manager = request.app.state.module_manager
    enabled_modules = [m for m in module_manager.get_all_modules() if m.get("enabled")]
    
    return templates.TemplateResponse(request, "index.html", {
        "modules": enabled_modules,
        "active_module": "chat",
        "sidebar_template": "chat_sidebar.html",
        "sessions": session_manager.list_sessions()
    })

@router.get("/gui", response_class=HTMLResponse)
async def chat_gui(request: Request, session_id: str = Query(None)):
    active_session = None
    if session_id:
        active_session = session_manager.get_session(session_id)
    
    if not active_session:
        # If no session specified or found, try to use the latest one or create one
        sessions = session_manager.list_sessions()
        if sessions:
            active_session = sessions[0]

    return templates.TemplateResponse(request, "chat_gui.html", {"session": active_session})

@router.get("/sessions", response_class=HTMLResponse)
async def get_chat_sessions(request: Request):
    sessions = session_manager.list_sessions()
    return templates.TemplateResponse(request, "chat_session_list.html", {"sessions": sessions})

@router.post("/sessions/new")
async def create_new_session():
    new_session = session_manager.create_session()
    import json
    trigger_data = {
        "sessionsChanged": None,
        "newSessionCreated": {"id": new_session['id']}
    }
    return HTMLResponse(content="", headers={"HX-Trigger": json.dumps(trigger_data)})

@router.post("/sessions/{session_id}/delete")
async def delete_session(request: Request, session_id: str):
    session_manager.delete_session(session_id)

    # Get the new chat GUI. By passing session_id=None, chat_gui will pick the latest session,
    # or create a new one if none are left.
    response = await chat_gui(request, session_id=None)

    # Add a trigger to the response to tell the session list to update itself.
    response.headers["HX-Trigger"] = "sessionsChanged"
    return response

@router.post("/sessions/{session_id}/rename")
async def rename_session(session_id: str, name: str = Form(...)):
    session_manager.rename_session(session_id, name)
    return HTMLResponse(content="", headers={"HX-Trigger": "sessionsChanged"})

@router.post("/send", response_class=HTMLResponse)
async def send_message(
    request: Request,
    message: str = Form(...),
    session_id: str = Query(None)
):
    if not session_id:
        return HTMLResponse("Error: No session selected", status_code=400)

    active_session = session_manager.get_session(session_id)
    if not active_session:
        return HTMLResponse("Error: Session not found", status_code=404)

    # Add user message to history
    session_manager.add_message(session_id, "user", message)
    # Reload session to get the updated history
    active_session = session_manager.get_session(session_id)
    
    # --- New AI Flow Execution Logic ---
    active_flow_id = settings.get("active_ai_flow")
    active_flow = flow_manager.get_flow(active_flow_id) if active_flow_id else None

    if not active_flow:
        ai_response = "Error: No active AI Flow is set. Please go to the AI Flow page to create and activate a flow."
    else:
        try:
            runner = FlowRunner(flow_id=active_flow['id'])
            
            # The initial input to the flow will be the full chat history.
            initial_data = {"messages": active_session["history"]}
            
            flow_result = await runner.run(initial_data)
            
            if "error" in flow_result:
                ai_response = f"Flow Execution Error: {flow_result['error']}"
            else:
                # The final output of the flow is expected to be the AI response content,
                # typically processed by a "Chat Output" node.
                ai_response = flow_result.get("content", "Flow finished but produced no valid response content.")
        except Exception as e:
            ai_response = f"Critical Error running AI Flow: {e}"

    # Add AI response to history
    session_manager.add_message(session_id, "assistant", ai_response)

    return templates.TemplateResponse(
        request, 
        "chat_message_pair.html", 
        {
            "user_message": message,
            "ai_response": ai_response
        }
    )
