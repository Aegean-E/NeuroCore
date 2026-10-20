"""Safe ZIP extraction for attacker-controlled marketplace packages.

A package uploaded to the marketplace is untrusted. A member named ``../evil.py`` (zip-slip), an
absolute path such as ``/etc/cron.d/x``, or a symlink would let the archive write anywhere the
process can reach. This module validates every member against the resolved destination before
writing, and refuses symlink entries outright.
"""

import os
import zipfile
from pathlib import PurePosixPath


class ArchiveSecurityError(Exception):
    """Raised when a ZIP member would escape the destination directory."""


def _is_symlink(info: zipfile.ZipInfo) -> bool:
    """Whether the member records a symbolic link in its Unix mode bits."""
    return (info.external_attr >> 16) & 0o170000 == 0o120000


def _safe_member_path(dest_dir: str, member: str) -> str:
    """The absolute write path for `member`, or raise when it escapes `dest_dir`."""
    normalized = PurePosixPath(member.replace("\\", "/"))
    if normalized.is_absolute() or normalized.drive:
        raise ArchiveSecurityError(f"archive member '{member}' has an absolute path")
    parts = [part for part in normalized.parts if part not in ("", ".")]
    if not parts or ".." in parts:
        raise ArchiveSecurityError(
            f"archive member '{member}' escapes the destination (zip-slip)")
    target = os.path.abspath(os.path.join(dest_dir, *parts))
    root = os.path.abspath(dest_dir)
    if target != root and not target.startswith(root + os.sep):
        raise ArchiveSecurityError(f"archive member '{member}' escapes the destination")
    return target


def _common_root(infos) -> str:
    """The single shared top-level directory of the archive as ``dir/``, or ``''``.

    A wrapped package puts every entry under one folder; unwrapping it lands the contents
    directly in the destination. Root-level files mean there is no single wrapper, so the
    members are kept as they are.
    """
    names = [info.filename for info in infos if info.filename and not info.filename.endswith("/")]
    if not names or any("/" not in name for name in names):
        return ""
    roots = {name.split("/", 1)[0] for name in names}
    if len(roots) == 1:
        root = next(iter(roots))
        # A traversal root ('..') or a drive-qualified root is not a wrapper directory; leaving
        # it in place lets the per-member check reject it instead of silently unwrapping it.
        if root not in (".", "..") and not PurePosixPath(root).drive:
            return root + "/"
    return ""


def extract_zip_safely(archive_path: str, dest_dir: str, strip_root: bool = False) -> list:
    """Extract `archive_path` into `dest_dir`, refusing escaping or symlink members.

    With `strip_root`, a single shared top-level directory is removed first. Returns the
    destination-relative paths written.
    """
    written = []
    with zipfile.ZipFile(archive_path) as archive:
        infos = archive.infolist()
        prefix = _common_root(infos) if strip_root else ""
        for info in infos:
            name = info.filename
            if prefix and name.startswith(prefix):
                name = name[len(prefix):]
            if not name or name.endswith("/"):
                continue
            if _is_symlink(info):
                raise ArchiveSecurityError(
                    f"archive member '{info.filename}' is a symbolic link")
            target = _safe_member_path(dest_dir, name)
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with archive.open(info) as source, open(target, "wb") as sink:  # hygiene: allow zip member binary stream
                sink.write(source.read())
            written.append(os.path.relpath(target, dest_dir).replace(os.sep, "/"))
    return written