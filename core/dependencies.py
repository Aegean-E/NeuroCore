from fastapi import Depends

from core.settings import SettingsManager, settings as settings_manager
from core.llm import LLMBridge


def get_settings_manager() -> SettingsManager:
    """Dependency to get the global settings manager instance."""
    return settings_manager


def get_llm_bridge(settings: SettingsManager = Depends(get_settings_manager)) -> LLMBridge:
    """Dependency to get a configured LLMBridge instance."""
    return LLMBridge(
        base_url=settings.get("llm_api_url"),
    )