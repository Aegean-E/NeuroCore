"""
Pytest configuration and fixtures for test isolation.

Provides fixtures to backup/restore files that are mutated in-place during tests,
ensuring test isolation and preventing cross-test contamination.
"""

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, List
import pytest


@pytest.fixture(autouse=True)
def reset_shared_async_state():
    """
    Reset module-level async singletons between tests.

    httpx.AsyncClient and asyncio.Lock are bound to the event loop that
    created them.  With asyncio_mode='auto', each test gets its own loop,
    so a client created in test N is invalid in test N+1.  Resetting the
    globals before (and after) each test forces lazy re-creation in the
    correct loop without touching the old loop's handles.

    We intentionally do NOT call client.aclose() here — doing so via
    loop.run_until_complete() during test teardown invalidates Win32 pipe
    handles that multiprocessing tests created later rely on.  The client
    reference is simply dropped; httpx will issue a ResourceWarning on GC
    but that is harmless in test runs.
    """
    import core.llm as llm_module
    from core.flow_runner import FlowRunner

    llm_module._shared_client = None
    llm_module._client_lock = None
    FlowRunner._cache_lock = None

    yield

    llm_module._shared_client = None
    llm_module._client_lock = None


@pytest.fixture(autouse=True)
def disable_debug_mode_for_tests():
    """
    Force debug_mode=False during every test.

    When debug_mode=True, FlowRunner._get_executor_class calls
    importlib.reload(node_dispatcher).  If that dispatcher imports from
    modules.tools.sandbox, Python creates a new function object for
    _execute_in_process that differs from the one already recorded in
    sys.modules['modules.tools.sandbox'].  multiprocessing.Process uses
    pickle to send the target function to the spawned worker; pickle
    resolves the function by module+qualname and raises PicklingError when
    the object identity doesn't match the live module attribute.

    Forcing debug_mode=False for the duration of each test prevents the
    reload entirely.  We also clear the executor class cache so the next
    test starts with a clean slate (no stale class from a previous reload).
    """
    from core.settings import settings
    from core.flow_runner import FlowRunner

    original_debug = settings.settings.get('debug_mode', False)
    settings.settings['debug_mode'] = False
    FlowRunner._executor_cache.clear()

    yield

    settings.settings['debug_mode'] = original_debug
    FlowRunner._executor_cache.clear()


# Files that need backup/restore during tests
PROTECTED_MODULE_FILES = [
    "modules/planner/module.json",
    "modules/reflection/module.json",
    "modules/chat/module.json",
    "modules/memory/module.json",
    "modules/knowledge_base/module.json",
]


class FileBackupManager:
    """Manages backup and restore operations for files during tests."""
    
    def __init__(self):
        self._backups: Dict[str, Optional[str]] = {}
        self._temp_dir: Optional[str] = None
    
    def backup_file(self, filepath: str) -> bool:
        """Back up a file if it exists."""
        if os.path.exists(filepath):
            self._backups[filepath] = filepath
            return True
        return False
    
    def restore_file(self, filepath: str) -> bool:
        """Restore a file from backup if one exists."""
        if filepath in self._backups:
            # File was backed up, restore it
            if self._backups[filepath] is not None:
                # Restore original content
                original_path = self._backups[filepath]
                if os.path.exists(original_path):
                    shutil.copy2(original_path, filepath)
                else:
                    # Original didn't exist, remove the modified file
                    if os.path.exists(filepath):
                        os.remove(filepath)
            return True
        return False
    
    def cleanup(self):
        """Clean up any temporary files."""
        # Clear all backups
        self._backups.clear()


# Global backup manager instance
_backup_manager = FileBackupManager()


@pytest.fixture(scope="session")
def backup_manager():
    """Session-scoped fixture to manage file backups across tests."""
    return _backup_manager


@pytest.fixture(autouse=True)
def protect_module_files():
    """
    Auto-use fixture that backs up module.json files before each test
    and restores them after each test, even if the test fails.
    
    This ensures test isolation by preventing modifications from persisting
    across test boundaries.
    """
    # Backup all protected module files before test
    backed_up_files = []
    for filepath in PROTECTED_MODULE_FILES:
        if os.path.exists(filepath):
            # Read original content
            with open(filepath, 'r') as f:
                original_content = f.read()
            _backup_manager._backups[filepath] = original_content
            backed_up_files.append(filepath)
    
    yield
    
    # Restore all backed up files after test (even on failure)
    for filepath in backed_up_files:
        if filepath in _backup_manager._backups:
            original_content = _backup_manager._backups[filepath]
            # Restore the original content
            with open(filepath, 'w') as f:
                f.write(original_content)
    
    # Clear the backups for this test
    for filepath in backed_up_files:
        _backup_manager._backups.pop(filepath, None)


@pytest.fixture
def temp_module_dir(tmp_path):
    """
    Provides a temporary directory for creating test module structures.
    
    Usage:
        def test_something(temp_module_dir):
            # temp_module_dir is a pathlib.Path to a temp directory
            ...
    """
    return tmp_path


@pytest.fixture
def sample_module_json():
    """Provides a sample module.json structure for testing."""
    return {
        "id": "test_module",
        "enabled": True,
        "order": 100,
        "config": {
            "test_key": "test_value"
        }
    }


@pytest.fixture
def backup_specific_file():
    """
    Fixture to backup a specific file before a test and restore it after.
    
    Usage:
        def test_something(backup_specific_file):
            filepath = "modules/planner/module.json"
            backup_specific_file(filepath)
            
            # Do test operations that modify the file
            
            # File is automatically restored after test
    """
    backed_files = []
    
    def _backup(filepath: str):
        """Back up a specific file."""
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                backed_files.append((filepath, f.read()))
    
    yield _backup
    
    # Restore all backed files
    for filepath, content in backed_files:
        with open(filepath, 'w') as f:
            f.write(content)


@pytest.fixture
def isolated_temp_dir(tmp_path):
    """
    Provides an isolated temporary directory with proper cleanup.
    
    Creates a subdirectory in the temp space and ensures proper cleanup
    even if the test fails.
    """
    test_dir = tmp_path / "isolated_test"
    test_dir.mkdir(exist_ok=True)
    yield test_dir
    # Cleanup happens automatically via tmp_path fixture


@pytest.fixture
def mock_settings_file(tmp_path, monkeypatch):
    """
    Creates a temporary settings file for testing settings functionality.
    
    Usage:
        def test_settings(mock_settings_file):
            # mock_settings_file is the path to temp settings.json
            ...
    """
    settings_file = tmp_path / "settings.json"
    
    # Set the file path in the module
    from core import settings as settings_module
    monkeypatch.setattr(settings_module, 'SETTINGS_FILE', str(settings_file))
    
    yield str(settings_file)
    
    # Cleanup is handled by tmp_path


@pytest.fixture
def module_json_backup():
    """
    Explicit fixture for backing up and restoring specific module.json files.
    
    Usage:
        def test_modify_planner_config(module_json_backup):
            # Backup specific files
            module_json_backup.backup("modules/planner/module.json")
            module_json_backup.backup("modules/reflection/module.json")
            
            # ... test code ...
            
            # Automatic restore happens after test
    """
    class ModuleBackup:
        def __init__(self):
            self._backups = {}
        
        def backup(self, filepath: str) -> bool:
            """Back up a module.json file."""
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    self._backups[filepath] = f.read()
                return True
            return False
        
        def restore(self, filepath: str):
            """Restore a specific file from backup."""
            if filepath in self._backups:
                with open(filepath, 'w') as f:
                    f.write(self._backups[filepath])
        
        def restore_all(self):
            """Restore all backed up files."""
            for filepath, content in self._backups.items():
                with open(filepath, 'w') as f:
                    f.write(content)
            self._backups.clear()
    
    backup = ModuleBackup()
    yield backup
    # Restore all after test
    backup.restore_all()

