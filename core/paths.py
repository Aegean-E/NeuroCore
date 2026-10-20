"""Absolute locations for NeuroCore runtime files.

Runtime state must never depend on the process working directory: starting the app or
its tests from a parent folder must not create settings, databases or caches there.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def project_path(*parts: str) -> str:
    """Return an absolute path inside the NeuroCore project root."""
    return str(PROJECT_ROOT.joinpath(*parts))
