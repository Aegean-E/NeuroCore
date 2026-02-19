import os
import json
from unittest.mock import MagicMock, patch
import pytest
from core.module_manager import ModuleManager

@pytest.fixture
def temp_modules_dir(tmp_path):
    """Creates a temporary module directory structure for testing."""
    modules_dir = tmp_path / "modules"
    modules_dir.mkdir()
    (modules_dir / "__init__.py").touch()  # Make it a package

    # Valid enabled module
    (modules_dir / "enabled_module").mkdir()
    with open(modules_dir / "enabled_module" / "module.json", "w") as f:
        json.dump({"name": "Enabled Module", "description": "Test", "enabled": True}, f)
    (modules_dir / "enabled_module" / "__init__.py").touch()
    (modules_dir / "enabled_module" / "router.py").write_text("from fastapi import APIRouter\nrouter = APIRouter()")

    # Valid disabled module
    (modules_dir / "disabled_module").mkdir()
    with open(modules_dir / "disabled_module" / "module.json", "w") as f:
        json.dump({"name": "Disabled Module", "description": "Test", "enabled": False}, f)

    # Invalid module (no json)
    (modules_dir / "invalid_module").mkdir()

    return modules_dir

def test_module_discovery(temp_modules_dir):
    """Tests that the manager correctly discovers modules with metadata."""
    manager = ModuleManager(modules_dir=str(temp_modules_dir))
    all_modules = manager.get_all_modules()
    
    assert len(all_modules) == 2
    module_names = {m['name'] for m in all_modules}
    assert "Enabled Module" in module_names
    assert "Disabled Module" in module_names

def test_load_enabled_modules(temp_modules_dir):
    """Tests that only enabled modules have their routers loaded."""
    manager = ModuleManager(modules_dir=str(temp_modules_dir))
    mock_app = MagicMock()
    
    # Create a mock module with a router attribute to be returned by the patch
    mock_router = MagicMock()
    mock_module = MagicMock()
    mock_module.router = mock_router
    
    # Patch importlib.import_module to avoid sys.path issues and return our mock
    with patch('importlib.import_module', return_value=mock_module) as mock_import:
        loaded_meta = manager.load_enabled_modules(mock_app)

    # Assert that import_module was called with the correct module name
    mock_import.assert_called_once_with('modules.enabled_module')
    
    # Assert that include_router was called once for the enabled module
    mock_app.include_router.assert_called_once_with(mock_router, prefix='/enabled_module', tags=['enabled_module'])
    
    # Assert that the correct metadata was returned
    assert len(loaded_meta) == 1
    assert loaded_meta[0]['name'] == "Enabled Module"

def test_enable_disable_module(temp_modules_dir):
    """Tests that toggling a module's state updates its JSON file."""
    manager = ModuleManager(modules_dir=str(temp_modules_dir))
    
    # Module starts disabled, let's enable it
    module_meta = manager.enable_module("disabled_module")
    assert module_meta['enabled'] is True

    # Verify the file on disk was changed
    with open(temp_modules_dir / "disabled_module" / "module.json", "r") as f:
        data = json.load(f)
    assert data['enabled'] is True

    # Now disable it
    module_meta = manager.disable_module("disabled_module")
    assert module_meta['enabled'] is False

    # Verify the file on disk was changed back
    with open(temp_modules_dir / "disabled_module" / "module.json", "r") as f:
        data = json.load(f)
    assert data['enabled'] is False

def test_toggle_nonexistent_module(temp_modules_dir):
    """Tests that toggling a non-existent module returns None."""
    manager = ModuleManager(modules_dir=str(temp_modules_dir))
    result = manager.enable_module("non_existent_module")
    assert result is None