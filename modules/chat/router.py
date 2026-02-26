from fastapi import APIRouter, Request, Form, Depends, Query, UploadFile, File
from fastapi.responses import HTMLResponse, Response
import base64
import time
from core.settings import settings
from core.dependencies import get_llm_bridge
from core.llm import LLMBridge
from modules.chat.sessions import session_manager
from fastapi.templating import Jinja2Templates
from core.flow_runner import FlowRunner
from core.flow_manager import flow_manager
import json

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
        "sessions": session_manager.list_sessions(),
        "settings": settings.settings
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

@router.post("/settings/save")
async def save_chat_settings(request: Request, auto_rename_turns: int = Form(...)):
    module_manager = request.app.state.module_manager
    chat_module = module_manager.modules.get("chat")
    if not chat_module:
        return Response(status_code=404)
        
    config = chat_module.get("config", {}).copy()
    config["auto_rename_turns"] = auto_rename_turns
    
    module_manager.update_module_config("chat", config)
    
    return Response(status_code=200, headers={"HX-Trigger": json.dumps({"showMessage": {"level": "success", "message": "Chat settings saved"}})})

@router.post("/send", response_class=HTMLResponse)
async def send_message(
    request: Request,
    message: str = Form(...),
    session_id: str = Query(None),
    image: UploadFile = File(None),
    llm: LLMBridge = Depends(get_llm_bridge)
):
    if not session_id:
        return HTMLResponse("Error: No session selected", status_code=400)

    active_session = session_manager.get_session(session_id)
    if not active_session:
        return HTMLResponse("Error: Session not found", status_code=404)

    # Prepare user content (Text or Multimodal)
    user_content = message
    if image and image.filename:
        contents = await image.read()
        encoded = base64.b64encode(contents).decode("utf-8")
        mime_type = image.content_type or "image/jpeg"
        image_url = f"data:{mime_type};base64,{encoded}"
        
        user_content = [
            {"type": "text", "text": message},
            {"type": "image_url", "image_url": {"url": image_url}}
        ]

    # Add user message to history
    session_manager.add_message(session_id, "user", user_content)
    active_session = session_manager.get_session(session_id)
    
    active_flow_ids = settings.get("active_ai_flows", [])
    active_flow = flow_manager.get_flow(active_flow_ids[0]) if active_flow_ids else None

    if not active_flow:
        ai_response = "Error: No active AI Flow is set. Please go to the AI Flow page to create and activate a flow."
        elapsed_time = 0
    else:
        start_time = time.time()
        try:
            runner = FlowRunner(flow_id=active_flow['id'])
            
            # The initial input to the flow will be the full chat history.
            initial_data = {"messages": active_session["history"]}
            
            flow_result = await runner.run(initial_data)
            
            elapsed_time = round(time.time() - start_time, 1)
            
            if "error" in flow_result:
                ai_response = f"Flow Execution Error: {flow_result['error']}"
            else:
                # The final output of the flow is expected to be the AI response content,
                # typically processed by a "Chat Output" node.
                ai_response = flow_result.get("content", "Flow finished but produced no valid response content.")
                
                # If the response indicates a failure, don't add it to history
                if "produced no valid response" in ai_response or "Error:" in ai_response:
                    ai_response = None  # Mark as failed so we don't add to history
        except Exception as e:
            ai_response = f"Critical Error running AI Flow: {e}"
            elapsed_time = round(time.time() - start_time, 1) if 'start_time' in locals() else 0

    # Add AI response to history only if it's valid
    if ai_response:
        session_manager.add_message(session_id, "assistant", ai_response)

    # Reload session to ensure we have the absolute latest state for auto-renaming
    active_session = session_manager.get_session(session_id)

    # --- Auto-Renaming Logic ---
    # Get config for auto-rename turns
    module_manager = request.app.state.module_manager
    chat_module = module_manager.modules.get("chat")
    config = chat_module.get("config", {}) if chat_module else {}
    auto_rename_turns = int(config.get("auto_rename_turns", 3))

    if len(active_session["history"]) >= auto_rename_turns * 2 and active_session["name"].startswith("Session "):
        try:
            # Construct a prompt to summarize the conversation
            # We use the first user message and the AI response
            user_text = active_session["history"][0]["content"] if isinstance(active_session["history"][0]["content"], str) else "Image/Multimodal Content"
            ai_text = active_session["history"][1]["content"]
            
            # Truncate to avoid huge context if messages are long
            summary_context = f"User: {user_text[:500]}\nAI: {ai_text[:500]}"
            
            prompt = f"Generate a short, concise title (3-5 words) for this conversation based on the start:\n\n{summary_context}\n\nTitle:"
            
            # Call LLM for title generation
            title_response = await llm.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=15
            )
            
            if "choices" in title_response:
                new_title = title_response["choices"][0]["message"]["content"].strip().strip('"')
                if new_title and len(new_title) < 50:
                    session_manager.rename_session(session_id, new_title)
        except Exception as e:
            print(f"Auto-rename failed: {e}")

    return templates.TemplateResponse(
        request, 
        "chat_message_pair.html", 
        {
            "user_message": user_content,
            "ai_response": ai_response or "Flow failed to produce a response. Please check the flow configuration.",
            "elapsed_time": elapsed_time
        },
        headers={"HX-Trigger": "sessionsChanged"}
    )
