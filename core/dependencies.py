from fastapi import Depends, Request

from core.settings import SettingsManager, settings as settings_manager
from core.llm import LLMBridge
from core.module_manager import ModuleManager


def get_settings_manager() -> SettingsManager:
    """Dependency to get the global settings manager instance."""
    return settings_manager


def get_llm_bridge(settings: SettingsManager = Depends(get_settings_manager)) -> LLMBridge:
    """Dependency to get a configured LLMBridge instance."""
    return LLMBridge(
        base_url=settings.get("llm_api_url"),
        api_key=settings.get("llm_api_key"),
        embedding_base_url=settings.get("embedding_api_url"),
        embedding_model=settings.get("embedding_model"),
    )

def get_module_manager(request: Request) -> ModuleManager:
    """Dependency to get the module manager instance from the app state."""
    return request.app.state.module_manager