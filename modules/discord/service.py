"""
DiscordService — singleton that manages the Discord bot lifecycle.

Mirrors TelegramService: runs the Gateway listener in a daemon thread,
maintains a per-(channel, user) → session mapping, and dispatches
each message through the active NeuroCore flow.
"""
import asyncio
import json
import logging
import os
import threading
import time

from .bridge import DiscordBridge
from .node import ConfigLoader
from modules.chat.sessions import session_manager
from core.flow_runner import FlowRunner
from core.settings import settings

logger = logging.getLogger(__name__)

SESSION_MAPPING_FILE = os.path.join(os.path.dirname(__file__), "discord_sessions.json")


class DiscordService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.bridge = None
            cls._instance.thread = None
            cls._instance.running = False
            cls._instance.session_map = cls._instance._load_session_map()
            cls._instance._loop = None
        return cls._instance

    # ------------------------------------------------------------------
    # Session map persistence
    # ------------------------------------------------------------------

    def _load_session_map(self) -> dict:
        if os.path.exists(SESSION_MAPPING_FILE):
            try:
                with open(SESSION_MAPPING_FILE, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                return {}
        return {}

    def _save_session_map(self) -> None:
        temp = SESSION_MAPPING_FILE + ".tmp"
        try:
            with open(temp, "w") as f:
                json.dump(self.session_map, f)
            os.replace(temp, SESSION_MAPPING_FILE)
        except (OSError, IOError):
            if os.path.exists(temp):
                try:
                    os.remove(temp)
                except OSError:
                    pass
            raise

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _get_or_create_loop(self) -> asyncio.AbstractEventLoop:
        if self._loop is None or self._loop.is_closed():
            self._loop = asyncio.new_event_loop()
        return self._loop

    def start(self) -> None:
        if self.running:
            return

        config = ConfigLoader.get_config()
        token = config.get("bot_token", "").strip()

        if not token:
            logger.warning("Discord Bridge: Bot token not set. Service paused.")
            return

        channel_id = config.get("channel_id", "")
        self.bridge = DiscordBridge(token, channel_id, log_fn=logger.info)
        self.running = True

        self.thread = threading.Thread(target=self._run_async_listen, daemon=True)
        self.thread.start()
        logger.info("Discord Bridge Service started.")

    def _run_async_listen(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(
                self.bridge.listen(self.handle_message, self.check_running)
            )
        finally:
            loop.close()

    def check_running(self) -> bool:
        if not ConfigLoader.is_enabled():
            self.running = False
        return self.running

    def stop(self) -> None:
        self.running = False
        if self.bridge:
            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(self.bridge.close())
            finally:
                loop.close()

    # ------------------------------------------------------------------
    # Message handling
    # ------------------------------------------------------------------

    def handle_message(self, msg: dict) -> None:
        try:
            loop = self._get_or_create_loop()
            loop.run_until_complete(self.process_message(msg))
        except Exception as e:
            logger.error(f"Error processing Discord message: {e}")

    async def process_message(self, msg: dict) -> None:
        # Ignore messages from bots (including ourselves)
        author = msg.get("author", {})
        if author.get("bot", False):
            return

        channel_id = msg.get("channel_id", "")
        user_id = author.get("id", "")
        username = author.get("username", "User")
        content = msg.get("content", "").strip()

        if not channel_id or not content:
            return

        # Honour the configured channel restriction
        config = ConfigLoader.get_config()
        allowed_channel = str(config.get("channel_id", "")).strip()
        if allowed_channel and channel_id != allowed_channel:
            return

        # ------------------------------------------------------------------
        # Built-in commands
        # ------------------------------------------------------------------

        if content == "/help":
            help_text = (
                "🤖 **NeuroCore Discord Bridge Commands**\n\n"
                "`/new_session` — Start a fresh conversation\n"
                "`/delete_session` — Delete current history\n"
                "`/help` — Show this message\n\n"
                "Just type to chat!"
            )
            await self.bridge.send_message(help_text, channel_id)
            return

        if content == "/new_session":
            session = session_manager.create_session(f"Discord {username}")
            self.session_map[f"{channel_id}:{user_id}"] = session["id"]
            self._save_session_map()
            await self.bridge.send_message("🆕 New session started.", channel_id)
            return

        if content == "/delete_session":
            map_key = f"{channel_id}:{user_id}"
            sess_id = self.session_map.get(map_key)
            if sess_id:
                session_manager.delete_session(sess_id)
                del self.session_map[map_key]
                self._save_session_map()
                await self.bridge.send_message("🗑️ Session deleted.", channel_id)
            else:
                await self.bridge.send_message("No active session to delete.", channel_id)
            return

        # ------------------------------------------------------------------
        # Normal chat — map channel+user → NeuroCore session
        # ------------------------------------------------------------------

        map_key = f"{channel_id}:{user_id}"
        sess_id = self.session_map.get(map_key)
        if not sess_id or not session_manager.get_session(sess_id):
            session = session_manager.create_session(f"Discord {username}")
            sess_id = session["id"]
            self.session_map[map_key] = sess_id
            self._save_session_map()

        session_manager.add_message(sess_id, "user", content)

        # ------------------------------------------------------------------
        # Run the active flow
        # ------------------------------------------------------------------

        active_flow_ids = settings.get("active_ai_flows", [])
        if not active_flow_ids:
            await self.bridge.send_message(
                "⚠️ No active AI Flow configured on server.", channel_id
            )
            return

        start_time = time.time()
        try:
            runner = FlowRunner(flow_id=active_flow_ids[0])
            session = session_manager.get_session(sess_id)
            initial_data = {
                "messages": session["history"],
                "_input_source": "discord",
            }

            result = await runner.run(initial_data)
            elapsed = round(time.time() - start_time, 1)

            if "error" in result:
                response_text = f"❌ Error: {result['error']}"
            elif "content" in result:
                response_text = result["content"]
            elif "choices" in result:
                try:
                    response_text = result["choices"][0]["message"]["content"]
                except (KeyError, IndexError, TypeError):
                    response_text = "Empty response."
            else:
                response_text = "Flow finished with no output."

            time_str = " (<1s)" if elapsed < 1 else f" ({elapsed}s)"
            response_text += time_str

            session_manager.add_message(sess_id, "assistant", response_text)
            await self.bridge.send_message(response_text, channel_id)

        except Exception as e:
            logger.error(f"Discord flow execution error: {e}")
            await self.bridge.send_message(f"❌ Internal Error: {e}", channel_id)


discord_service = DiscordService()
