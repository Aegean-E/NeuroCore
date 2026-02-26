import json
import os
import asyncio
import threading
import base64
import time
from .bridge import TelegramBridge
from .node import ConfigLoader
from modules.chat.sessions import session_manager
from core.flow_runner import FlowRunner
from core.settings import settings

SESSION_MAPPING_FILE = os.path.join(os.path.dirname(__file__), "telegram_sessions.json")

class TelegramService:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TelegramService, cls).__new__(cls)
            cls._instance.bridge = None
            cls._instance.thread = None
            cls._instance.running = False
            cls._instance.session_map = cls._instance._load_session_map()
        return cls._instance

    def _load_session_map(self):
        if os.path.exists(SESSION_MAPPING_FILE):
            try:
                with open(SESSION_MAPPING_FILE, 'r') as f:
                    return json.load(f)
            except: return {}
        return {}

    def _save_session_map(self):
        with open(SESSION_MAPPING_FILE, 'w') as f:
            json.dump(self.session_map, f)

    def start(self):
        if self.running: return
        
        config = ConfigLoader.get_config()
        token = config.get("bot_token")
        
        if not token:
            print("Telegram Bridge Token not set. Service paused.")
            return

        # Initialize bridge with dummy chat_id for listening; send_message will override it
        self.bridge = TelegramBridge(token, 0, log_fn=print)
        self.running = True
        
        self.thread = threading.Thread(target=self.bridge.listen, args=(self.handle_message, self.check_running))
        self.thread.daemon = True
        self.thread.start()
        print("Telegram Bridge Service Started")

    def check_running(self):
        # Check if module is still enabled in file
        if not ConfigLoader.is_enabled():
            self.running = False
        return self.running

    def handle_message(self, msg):
        try:
            # Run async process_message in a new event loop for this thread
            asyncio.run(self.process_message(msg))
        except Exception as e:
            print(f"Error processing Telegram message: {e}")

    async def process_message(self, msg):
        chat_id = msg.get("chat_id")
        msg_type = msg.get("type")
        text = msg.get("text", "").strip()
        caption = msg.get("caption", "").strip()
        
        # Determine command text (text message or caption)
        command_text = text if text else caption
        
        if not chat_id: return
        
        # Only process text or photo messages for now
        if msg_type not in ["text", "photo"]:
            return

        # --- Commands ---
        if command_text == "/help":
            help_text = (
                "🤖 *NeuroCore Bridge Commands*\n\n"
                "/new_session - Start a fresh conversation\n"
                "/delete_session - Delete current history\n"
                "/help - Show this message\n\n"
                "Just type to chat! Send images to use vision capabilities."
            )
            self.bridge.send_message(help_text, chat_id)
            return

        if command_text == "/new_session":
            session = session_manager.create_session(f"Telegram {chat_id}")
            self.session_map[str(chat_id)] = session["id"]
            self._save_session_map()
            self.bridge.send_message("🆕 New session started.", chat_id)
            return

        if command_text == "/delete_session":
            sess_id = self.session_map.get(str(chat_id))
            if sess_id:
                session_manager.delete_session(sess_id)
                del self.session_map[str(chat_id)]
                self._save_session_map()
                self.bridge.send_message("🗑️ Session deleted.", chat_id)
            else:
                self.bridge.send_message("No active session to delete.", chat_id)
            return

        # --- Normal Chat Flow ---
        
        # 1. Get or Create Session
        sess_id = self.session_map.get(str(chat_id))
        if not sess_id or not session_manager.get_session(sess_id):
            session = session_manager.create_session(f"Telegram {chat_id}")
            sess_id = session["id"]
            self.session_map[str(chat_id)] = sess_id
            self._save_session_map()

        # 2. Prepare User Content
        user_content = None
        
        if msg_type == "text":
            if not text: return
            user_content = text
            
        elif msg_type == "photo":
            photo_data = msg.get("photo")
            if not photo_data: return
            
            file_id = photo_data.get("file_id")
            try:
                file_info = self.bridge.get_file_info(file_id)
                file_path_remote = file_info.get("file_path")
                
                if file_path_remote:
                    # Create temp path
                    temp_dir = os.path.join("temp", "telegram")
                    os.makedirs(temp_dir, exist_ok=True)
                    local_filename = f"{file_id}_{os.path.basename(file_path_remote)}"
                    local_path = os.path.join(temp_dir, local_filename)
                    
                    # Run blocking download in a separate thread to avoid freezing the event loop
                    loop = asyncio.get_running_loop()
                    if await loop.run_in_executor(None, self.bridge.download_file, file_path_remote, local_path):
                        with open(local_path, "rb") as img_file:
                            b64_data = base64.b64encode(img_file.read()).decode('utf-8')
                        
                        # Cleanup
                        try:
                            os.remove(local_path)
                        except: pass
                        
                        # Construct Multimodal Message
                        user_content = []
                        if caption:
                            user_content.append({"type": "text", "text": caption})
                        
                        # Determine mime type
                        mime_type = "image/jpeg"
                        if file_path_remote.lower().endswith(".png"):
                            mime_type = "image/png"
                        elif file_path_remote.lower().endswith(".webp"):
                            mime_type = "image/webp"
                            
                        user_content.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime_type};base64,{b64_data}"}
                        })
                    else:
                        self.bridge.send_message("⚠️ Failed to download image.", chat_id)
                        return
                else:
                    self.bridge.send_message("⚠️ Could not retrieve file info.", chat_id)
                    return
            except Exception as e:
                print(f"Telegram image error: {e}")
                self.bridge.send_message("⚠️ Error processing image.", chat_id)
                return

        if not user_content: return

        # 3. Add User Message
        session_manager.add_message(sess_id, "user", user_content)
        
        # 4. Run Active Flow
        active_flow_ids = settings.get("active_ai_flows", [])
        if not active_flow_ids:
            self.bridge.send_message("⚠️ No active AI Flow configured on server.", chat_id)
            return

        active_flow_id = active_flow_ids[0]
        start_time = time.time()
        try:
            runner = FlowRunner(flow_id=active_flow_id)
            session = session_manager.get_session(sess_id)
            initial_data = {"messages": session["history"], "_input_source": "telegram"}
            
            result = await runner.run(initial_data)
            
            elapsed_time = round(time.time() - start_time, 1)
            
            response_text = ""
            if "error" in result:
                response_text = f"❌ Error: {result['error']}"
            elif "content" in result:
                response_text = result["content"]
            elif "choices" in result: 
                 try:
                    response_text = result["choices"][0]["message"]["content"]
                 except: response_text = "Empty response."
            else:
                 response_text = "Flow finished with no output."

            # Add response time to the message
            time_str = f" (<1s)" if elapsed_time < 1 else f" ({elapsed_time}s)"
            response_text += time_str

            # 5. Add Assistant Message & Reply
            session_manager.add_message(sess_id, "assistant", response_text)
            self.bridge.send_message(response_text, chat_id)

        except Exception as e:
            print(f"Flow execution error: {e}")
            self.bridge.send_message(f"❌ Internal Error: {e}", chat_id)

telegram_service = TelegramService()