"""Runtime state must stay inside the project, whatever the launch directory is."""

import os
import subprocess
import sys
from pathlib import Path

from core.flow_manager import FLOWS_FILE
from core.module_manager import MODULES_DIR
from core.observability import get_trace_file_path
from core.paths import PROJECT_ROOT
from core.session_manager import EPISODE_DIR, SESSION_FILE, TRACE_FILE
from core.settings import SETTINGS_FILE
from modules.calendar.events import EVENTS_FILE
from modules.chat.sessions import SESSIONS_FILE
from modules.reasoning_book.service import DATA_FILE

RUNTIME_PATHS = (
    SETTINGS_FILE,
    FLOWS_FILE,
    MODULES_DIR,
    SESSION_FILE,
    TRACE_FILE,
    EPISODE_DIR,
    get_trace_file_path(),
    EVENTS_FILE,
    SESSIONS_FILE,
    DATA_FILE,
)


def test_runtime_paths_are_absolute_and_inside_project():
    for runtime_path in RUNTIME_PATHS:
        path = Path(runtime_path)
        assert path.is_absolute(), runtime_path
        assert path == PROJECT_ROOT or PROJECT_ROOT in path.parents, runtime_path


def test_importing_the_app_from_another_directory_writes_nothing_there(tmp_path):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, "-c", "import main"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert list(tmp_path.iterdir()) == []
