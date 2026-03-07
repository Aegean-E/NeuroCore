import asyncio
import copy
import json
import os
import uuid
from datetime import datetime
import threading

SESSIONS_FILE = "chat_sessions.json"


def _estimate_tokens(messages: list) -> int:
    """
    Rough token estimate using the ~4 chars/token approximation.
    Works for any LLM provider without extra dependencies.
    Images are counted as ~500 tokens each.
    """
    total = 0
    for m in messages:
        content = m.get("content", "")
        if isinstance(content, list):
            for part in content:
                if part.get("type") == "text":
                    total += len(part.get("text", "")) // 4
                else:
                    total += 500  # rough estimate for images/media
        else:
            total += len(str(content)) // 4
    return total

class SessionManager:
    def __init__(self, storage_file=SESSIONS_FILE):
        self.storage_file = storage_file
        self._sync_lock = threading.Lock()  # For sync methods
        self._async_lock = asyncio.Lock()   # For async methods
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
        """Save sessions to disk using atomic temp-file-and-rename pattern."""
        temp_path = self.storage_file + ".tmp"
        try:
            with open(temp_path, "w") as f:
                json.dump(self.sessions, f, indent=4)
            # Atomic replace on all platforms
            os.replace(temp_path, self.storage_file)
        except Exception as e:
            # Clean up temp file on failure
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise e

    def create_session(self, name=None):
        with self._sync_lock:
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
        with self._sync_lock:
            session = self.sessions.get(session_id)
            return copy.deepcopy(session) if session is not None else None

    def delete_session(self, session_id):
        with self._sync_lock:
            if session_id in self.sessions:
                del self.sessions[session_id]
                self._save_sessions()
                return True
            return False

    def rename_session(self, session_id, new_name):
        with self._sync_lock:
            if session_id in self.sessions:
                self.sessions[session_id]["name"] = new_name
                self._save_sessions()
                return True
            return False

    def list_sessions(self):
        with self._sync_lock:
            return sorted(self.sessions.values(), key=lambda x: x.get('updated_at', x['created_at']), reverse=True)

    def add_message(self, session_id, role, content):
        with self._sync_lock:
            if session_id in self.sessions:
                self.sessions[session_id]["history"].append({"role": role, "content": content})
                self.sessions[session_id]["updated_at"] = datetime.now().isoformat()
                self._save_sessions()
                return True
            return False

    async def compact_session(self, session_id: str, llm_bridge, keep_last: int = 10):
        """
        Summarize old messages and replace them with a compact summary system message.

        The LLM call is made outside the lock to avoid blocking other operations.
        History is only updated (under lock) after a successful summarization.

        Returns:
            (compacted: bool, tokens_before: int)
        """
        # Snapshot history under lock to avoid TOCTOU race
        async with self._async_lock:
            session = self.sessions.get(session_id)
            if not session:
                return False, 0
            session_copy = copy.deepcopy(session)
            snapshot_updated_at = session_copy['updated_at']
            history = session_copy["history"]

        tokens_before = _estimate_tokens(history)

        # Need at least keep_last + 2 messages to be worth compacting
        if len(history) <= keep_last + 2:
            return False, tokens_before

        old_messages = history[:-keep_last]
        recent_messages = history[-keep_last:]

        # Build a readable transcript of the old messages for the LLM to summarize
        conversation_lines = []
        for m in old_messages:
            role = m["role"].upper()
            content = m["content"]
            if isinstance(content, list):
                # Multimodal — extract text parts only
                text_parts = [p.get("text", "") for p in content if p.get("type") == "text"]
                content = " ".join(text_parts) or "[Image/Media]"
            # Truncate very long individual messages to keep the prompt manageable
            conversation_lines.append(f"{role}: {str(content)[:800]}")

        summary_prompt = [
            {
                "role": "user",
                "content": (
                    "Summarize the following conversation history concisely in 3-5 sentences. "
                    "Capture the key topics, decisions, facts, and context that would be "
                    "important for continuing the conversation:\n\n"
                    + "\n".join(conversation_lines)
                )
            }
        ]

        try:
            result = await llm_bridge.chat_completion(
                messages=summary_prompt,
                max_tokens=400,
                temperature=0.3
            )
            summary_text = result["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"[Compact] LLM summarization failed: {e}")
            return False, tokens_before

        # Replace history: one summary system message + the most recent messages verbatim
        new_history = [
            {
                "role": "system",
                "content": f"[Conversation Summary — earlier messages compacted]: {summary_text}"
            }
        ] + recent_messages

        # Re-validate timestamp before writing back to avoid race condition
        async with self._async_lock:
            current = self.sessions.get(session_id)
            if current and current['updated_at'] == snapshot_updated_at:
                self.sessions[session_id]["history"] = new_history
                self.sessions[session_id]["updated_at"] = datetime.now().isoformat()
                # Use asyncio.to_thread to avoid blocking event loop with sync I/O
                await asyncio.to_thread(self._save_sessions)
                return True, tokens_before

        return False, tokens_before


session_manager = SessionManager()
