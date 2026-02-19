import os
import json
from unittest.mock import patch
import pytest
from fastapi.testclient import TestClient
from core.module_manager import ModuleManager


@pytest.fixture(scope="module")
def temp_modules_dir(tmp_path_factory):
    """Creates a temporary module directory for the test session."""
    modules_dir = tmp_path_factory.mktemp("modules_api")
    (modules_dir / "__init__.py").touch()

    # Create a test module
    (modules_dir / "toggle_test").mkdir()
    with open(modules_dir / "toggle_test" / "module.json", "w") as f:
        json.dump({"name": "Toggle Test", "description": "A test module.", "enabled": False, "id": "toggle_test"}, f)

    return str(modules_dir)


@pytest.fixture(scope="module")
def client(temp_modules_dir):
    """A TestClient that uses a temporary module directory."""
    # Patch the ModuleManager instance in main to use our temp directory
    with patch('main.module_manager', new=ModuleManager(modules_dir=temp_modules_dir)):
        from main import app
        with TestClient(app) as c:
            yield c


def test_get_modules_page(client):
    """Tests if the module management page loads correctly."""
    response = client.get("/modules")
    assert response.status_code == 200
    assert "Module Management" in response.text
    assert "Toggle Test" in response.text
    assert "Disabled" in response.text


def test_enable_module_route(client, temp_modules_dir):
    """Tests the API endpoint for enabling a module."""
    response = client.post("/modules/toggle_test/enable", follow_redirects=False)
    assert response.status_code == 303  # Redirect

    with open(os.path.join(temp_modules_dir, "toggle_test", "module.json"), "r") as f:
        data = json.load(f)
    assert data['enabled'] is True


def test_disable_module_route(client, temp_modules_dir):
    """Tests the API endpoint for disabling a module."""
    # First, ensure it's enabled
    client.post("/modules/toggle_test/enable")

    # Now, disable it
    response = client.post("/modules/toggle_test/disable", follow_redirects=False)
    assert response.status_code == 303  # Redirect

    with open(os.path.join(temp_modules_dir, "toggle_test", "module.json"), "r") as f:
        data = json.load(f)
    assert data['enabled'] is False