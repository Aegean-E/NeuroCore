from unittest.mock import MagicMock
import pytest
from core.dependencies import get_llm_bridge
from core.llm import LLMBridge

def test_get_llm_bridge_uses_settings():
    """
    Tests that the get_llm_bridge dependency correctly uses the
    SettingsManager to configure the LLMBridge instance.
    """
    # Create a mock SettingsManager
    mock_settings = MagicMock()
    mock_settings.get.return_value = "http://mock-url.com/v1"

    # Call the dependency function with the mock
    bridge = get_llm_bridge(settings=mock_settings)

    # Assert that the settings were used correctly
    mock_settings.get.assert_called_once_with("llm_api_url")
    
    # Assert that the bridge was created with the correct URL
    assert isinstance(bridge, LLMBridge)
    assert bridge.base_url == "http://mock-url.com/v1"