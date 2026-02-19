from fastapi import APIRouter, Request, Form
from fastapi.responses import HTMLResponse
from core.llm import llm

router = APIRouter()

@router.get("/gui", response_class=HTMLResponse)
async def chat_gui(request: Request):
    return """
    <div class="flex flex-col h-full">
        <div class="border-b border-slate-800 pb-4 mb-4 flex justify-between items-center">
            <div>
                <h2 class="text-xl font-semibold">AI Assistant</h2>
                <p class="text-sm text-slate-500">Connected to LM Studio</p>
            </div>
        </div>
        
        <div id="chat-messages" class="flex-grow space-y-4 mb-6 custom-scrollbar overflow-y-auto pr-2">
            <div class="bg-slate-800 p-4 rounded-2xl rounded-tl-none max-w-[80%]">
                <p class="text-sm">Hello! I'm NeuroCore. How can I assist you today?</p>
            </div>
        </div>
        
        <form hx-post="/chat/send" hx-target="#chat-messages" hx-swap="beforeend" hx-on::after-request="this.reset()" class="flex space-x-2">
            <input type="text" name="message" placeholder="Type your message..." 
                   class="flex-grow bg-slate-900 border border-slate-800 rounded-xl px-4 py-3 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 transition-all">
            <button type="submit" class="bg-blue-600 hover:bg-blue-500 text-white px-6 py-3 rounded-xl text-sm font-semibold transition-all">
                Send
            </button>
        </form>
    </div>
    """

@router.post("/send", response_class=HTMLResponse)
async def send_message(message: str = Form(...)):
    # Call LM Studio API
    messages = [{"role": "user", "content": message}]
    response = await llm.chat_completion(messages)
    
    if "error" in response:
        ai_response = f"Error connecting to LM Studio: {response['error']}"
    else:
        ai_response = response.get("choices", [{}])[0].get("message", {}).get("content", "No response.")

    return f"""
    <div class="flex flex-col items-end space-y-1">
        <div class="bg-blue-600 p-4 rounded-2xl rounded-tr-none max-w-[80%] text-white">
            <p class="text-sm">{message}</p>
        </div>
    </div>
    <div class="bg-slate-800 p-4 rounded-2xl rounded-tl-none max-w-[80%] mt-4 animate-in fade-in slide-in-from-left-2 duration-300">
        <div class="text-sm prose prose-invert">{ai_response}</div>
    </div>
    """
