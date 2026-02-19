import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_settings_page_accessible():
    response = client.get("/settings")
    assert response.status_code == 200
    assert "General Settings" in response.text
    assert 'name="llm_api_url"' in response.text


def test_save_settings_route():
    payload = {
        "llm_api_url": "http://new-test:1234/v1",
        "default_model": "new-model",
        "temperature": 0.8,
        "max_tokens": 512
    }
    # Test POST and Redirect
    response = client.post("/settings/save", data=payload, follow_redirects=False)
    assert response.status_code == 200

    # Verify the settings were actually saved (optional but good)
    from core.settings import settings
    settings.load_settings() # Reload from file
    assert settings.get("llm_api_url") == "http://new-test:1234/v1"
    assert settings.get("temperature") == 0.8