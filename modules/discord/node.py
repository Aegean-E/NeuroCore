import json
import logging
from .bridge import DiscordBridge

logger = logging.getLogger(__name__)


class ConfigLoader:
    @staticmethod
    def get_config() -> dict:
        try:
            with open("modules/discord/module.json", "r") as f:
                return json.load(f).get("config", {})
        except (json.JSONDecodeError, OSError, KeyError) as e:
            logger.warning(f"Failed to load discord config: {e}")
            return {}

    @staticmethod
    def is_enabled() -> bool:
        try:
            with open("modules/discord/module.json", "r") as f:
                return json.load(f).get("enabled", False)
        except (json.JSONDecodeError, OSError, KeyError) as e:
            logger.warning(f"Failed to check discord enabled status: {e}")
            return False


class DiscordInputExecutor:
    """Pass-through node that gates Repeater triggers."""

    async def receive(self, input_data: dict, config: dict = None) -> dict:
        if input_data.get("_repeat_count", 0) > 0:
            return None
        return input_data

    async def send(self, processed_data: dict) -> dict:
        if processed_data and processed_data.get("_repeat_count", 0) > 0:
            return processed_data
        if "messages" not in processed_data and "text" not in processed_data:
            return {
                "error": (
                    "Flow started without 'messages'. "
                    "'Discord Input' node requires it."
                )
            }
        return processed_data


class DiscordOutputExecutor:
    """Sends the flow's output ``content`` to the configured Discord channel."""

    _bridge_cache: dict = {}

    async def receive(self, input_data: dict, config: dict = None) -> dict:
        if not input_data or "content" not in input_data:
            return input_data

        mod_config = ConfigLoader.get_config()
        bot_token = mod_config.get("bot_token", "").strip()
        channel_id = str(mod_config.get("channel_id", "")).strip()

        if bot_token and channel_id:
            cache_key = f"{bot_token}_{channel_id}"
            if cache_key not in self._bridge_cache:
                self._bridge_cache[cache_key] = DiscordBridge(bot_token, channel_id)
            bridge = self._bridge_cache[cache_key]
            await bridge.send_message(input_data["content"])

        return input_data

    async def send(self, processed_data: dict) -> dict:
        return processed_data


async def get_executor_class(node_type_id: str):
    if node_type_id == "discord_input":
        return DiscordInputExecutor
    if node_type_id == "discord_output":
        return DiscordOutputExecutor
    return None
