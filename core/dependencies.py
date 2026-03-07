from fastapi import Depends, Request

from core.settings import SettingsManager, settings as settings_manager
from core.llm import LLMBridge
from core.module_manager import ModuleManager


# Module-level cached LLM bridge instance
_llm_bridge_instance: LLMBridge = None


def get_settings_manager() -> SettingsManager:
    """Dependency to get the global settings manager instance."""
    return settings_manager


def get_llm_bridge(settings: SettingsManager = Depends(get_settings_manager)) -> LLMBridge:
    """Dependency to get a cached LLMBridge instance.
    
    Uses a module-level singleton to avoid creating a new LLMBridge
    on every request. This is safe because:
    1. The settings values (URLs, API keys) are read at request time
    2. The underlying httpx.AsyncClient is already shared via get_shared_client()
    3. Only the bridge wrapper object is cached, not the settings values
    """
    global _llm_bridge_instance
    
    # Only create instance if it doesn't exist
    if _llm_bridge_instance is None:
        _llm_bridge_instance = LLMBridge(
            base_url=settings.get("llm_api_url"),
            api_key=settings.get("llm_api_key"),
            embedding_base_url=settings.get("embedding_api_url"),
            embedding_model=settings.get("embedding_model"),
            timeout=float(settings.get("request_timeout", 60.0)),
        )
    else:
        # Update settings in case they changed at runtime
        _llm_bridge_instance.base_url = settings.get("llm_api_url")
        _llm_bridge_instance.api_key = settings.get("llm_api_key")
        _llm_bridge_instance.embedding_base_url = settings.get("embedding_api_url") or _llm_bridge_instance.base_url
        _llm_bridge_instance.embedding_model = settings.get("embedding_model")
        _llm_bridge_instance.timeout = float(settings.get("request_timeout", 60.0))
    
    return _llm_bridge_instance


def get_module_manager(request: Request) -> ModuleManager:
    """Dependency to get the module manager instance from the app state."""
    return request.app.state.module_manager
