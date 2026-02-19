import json
import os
import uuid
from datetime import datetime

SESSIONS_FILE = "chat_sessions.json"

class SessionManager:
    def __init__(self, storage_file=SESSIONS_FILE):
        self.storage_file = storage_file
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
            json.dump(self.sessions, f, indent=4)

    def create_session(self, name=None):
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = {
            "id": session_id,
            "name": name or f"Session {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "history": [],
            "created_at": datetime.now().isoformat()
        }
        self._save_sessions()
        return self.sessions[session_id]

    def get_session(self, session_id):
        return self.sessions.get(session_id)

    def delete_session(self, session_id):
        if session_id in self.sessions:
            del self.sessions[session_id]
            self._save_sessions()
            return True
        return False

    def rename_session(self, session_id, new_name):
        if session_id in self.sessions:
            self.sessions[session_id]["name"] = new_name
            self._save_sessions()
            return True
        return False

    def list_sessions(self):
        # Return sorted by creation date (newest first)
        return sorted(self.sessions.values(), key=lambda x: x['created_at'], reverse=True)

    def add_message(self, session_id, role, content):
        if session_id in self.sessions:
            self.sessions[session_id]["history"].append({"role": role, "content": content})
            self._save_sessions()
            return True
        return False

session_manager = SessionManager()
