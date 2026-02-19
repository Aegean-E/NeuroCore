import requests
import os
from datetime import datetime
import time
import threading
from typing import List, Dict, Callable
import logging

BASE_URL = "https://api.telegram.org/bot"

class TelegramBridge:
    """Handles communication with Telegram API"""

    def __init__(self, bot_token: str, chat_id: int, log_fn: Callable[[str], None] = print):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.log = log_fn
        self.is_connected = False
        self.last_update_id = None
        self.session = requests.Session()
        self._stop_event = threading.Event()

    def stop(self):
        """Signal the listener to stop."""
        self._stop_event.set()

    def close(self):
        """Close the session."""
        self.stop()
        if self.session:
            self.session.close()

    def _send_message_chunk(self, text, chat_id=None) -> bool:
        try:
            target_chat_id = chat_id if chat_id is not None else self.chat_id
            url = f"{BASE_URL}{self.bot_token}/sendMessage"
            response = self.session.post(url, json={
                "chat_id": target_chat_id,
                "text": text
            }, timeout=30)
            response.raise_for_status()
            return True
        except Exception as e:
            self.log(f"⚠️ Chunk send failed: {e}")
            return False

    def send_message(self, text: str, chat_id: int = None) -> bool:
        """Send message to Telegram"""
        try:
            if not text: return False
            limit = 3072
            all_success = True
            for i in range(0, len(text), limit):
                chunk = text[i:i + limit]
                if not self._send_message_chunk(chunk, chat_id):
                    all_success = False
            return all_success
        except Exception as e:
            self.log(f"❌ Telegram send error: {e}")
            return False

    def get_messages(self) -> List[Dict]:
        """Get new messages from Telegram"""
        try:
            # Use offset + 1 to confirm processed messages to Telegram
            offset = self.last_update_id + 1 if self.last_update_id is not None else None
            
            url = f"{BASE_URL}{self.bot_token}/getUpdates"
            params = {"timeout": 40}
            if offset is not None: params["offset"] = offset
            
            response = self.session.get(url, params=params, timeout=45)
            response.raise_for_status()
            updates = response.json()
            
            messages = []

            for update in updates.get("result", []):
                # Update last_update_id to the current update's ID
                self.last_update_id = update["update_id"]

                if "message" in update:
                    msg = update["message"]
                    
                    # Base message data
                    message_data = {
                        "id": update["update_id"],
                        "chat_id": msg.get("chat", {}).get("id"),
                        "from": msg.get("from", {}).get("first_name", "Unknown"),
                        "timestamp": datetime.now().isoformat(),
                        "date": msg.get("date", int(time.time())), # Capture Telegram timestamp
                        "type": "unknown"
                    }

                    if "text" in msg:
                        message_data["type"] = "text"
                        message_data["text"] = msg["text"]
                    elif "document" in msg:
                        message_data["type"] = "document"
                        message_data["document"] = msg["document"]
                    elif "photo" in msg:
                        message_data["type"] = "photo"
                        # Telegram sends multiple sizes; take the last one (highest quality)
                        message_data["photo"] = msg["photo"][-1]
                        message_data["caption"] = msg.get("caption", "")
                    elif "voice" in msg:
                        message_data["type"] = "voice"
                        message_data["voice"] = msg["voice"]
                    
                    if message_data["type"] != "unknown":
                        messages.append(message_data)
            return messages
        except Exception as e:
            self.log(f"❌ Telegram receive error: {e}")
            return []

    def get_file_info(self, file_id: str) -> Dict:
        """Get file metadata from Telegram"""
        url = f"{BASE_URL}{self.bot_token}/getFile"
        response = self.session.get(url, params={"file_id": file_id}, timeout=30)
        response.raise_for_status()
        return response.json()['result']

    def download_file(self, file_path: str, save_path: str, max_size_mb: int = 20) -> bool:
        """Download file from Telegram with size limit"""
        url = f"https://api.telegram.org/file/bot{self.bot_token}/{file_path}"
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        
        try:
            response = self.session.get(url, timeout=120, stream=True)
            response.raise_for_status()

            # Check Content-Length header
            content_length = int(response.headers.get('Content-Length', 0))
            if content_length > max_size_mb * 1024 * 1024:
                self.log(f"⚠️ Telegram file too large ({content_length} bytes)")
                return False

            downloaded = 0
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if downloaded > max_size_mb * 1024 * 1024:
                        self.log(f"⚠️ Telegram file too large (streaming check)")
                        break

            if downloaded > max_size_mb * 1024 * 1024:
                if os.path.exists(save_path):
                    os.remove(save_path)
                return False

            return True
        except Exception as e:
            self.log(f"❌ Telegram download error: {e}")
            if os.path.exists(save_path):
                os.remove(save_path)
            return False

    def listen(self, 
               on_message: Callable[[Dict], None], 
               running_check: Callable[[], bool]):
        """
        Poll for messages and dispatch to callback.
        """
        self.log("🔌 Telegram Bridge: Listening for messages...")
        self._stop_event.clear()

        while running_check() and not self._stop_event.is_set():
            try:
                messages = self.get_messages()
                for msg in messages:
                    on_message(msg)
                
                self._stop_event.wait(1.0)
            except Exception as e:
                self.log(f"Error polling messages: {e}")
                self._stop_event.wait(1.0)
        self.log("🔌 Telegram Bridge: Stopped listening.")