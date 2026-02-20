import json
import os
import uuid
from datetime import datetime
import threading

SESSIONS_FILE = "chat_sessions.json"

class SessionManager:
    def __init__(self, storage_file=SESSIONS_FILE):
        self.storage_file = storage_file
        self.lock = threading.Lock()
        self.sessions = self._load_sessions()

    def _load_sessions(self):
        if os.path.exists(self.storage_file):
            with open(self.storage_file, "r") as f:
                try:
                    return json.load(f)
                except json.JSONDecodeError:
                    return {}
        return {}

    def _save_sessions(self):
        with open(self.storage_file, "w") as f:
            # Ensure atomic write by locking during dump
            # (Though the lock is usually held by the caller method)
            json.dump(self.sessions, f, indent=4)

    def create_session(self, name=None):
        with self.lock:
            session_id = str(uuid.uuid4())
            now = datetime.now().isoformat()
            self.sessions[session_id] = {
                "id": session_id,
                "name": name or f"Session {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                "history": [],
                "created_at": now,
                "updated_at": now
            }
            self._save_sessions()
            return self.sessions[session_id]

    def get_session(self, session_id):
        with self.lock:
            return self.sessions.get(session_id)

    def delete_session(self, session_id):
        with self.lock:
            if session_id in self.sessions:
                del self.sessions[session_id]
                self._save_sessions()
                return True
            return False

    def rename_session(self, session_id, new_name):
        with self.lock:
            if session_id in self.sessions:
                self.sessions[session_id]["name"] = new_name
                self._save_sessions()
                return True
            return False

    def list_sessions(self):
        with self.lock:
            return sorted(self.sessions.values(), key=lambda x: x.get('updated_at', x['created_at']), reverse=True)

    def add_message(self, session_id, role, content):
        with self.lock:
            if session_id in self.sessions:
                self.sessions[session_id]["history"].append({"role": role, "content": content})
                self.sessions[session_id]["updated_at"] = datetime.now().isoformat()
                self._save_sessions()
                return True
            return False

session_manager = SessionManager()
