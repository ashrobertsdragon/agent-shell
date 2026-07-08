"""Hardening helpers for on-disk history files.

Command history may contain inline secrets (API keys, bearer tokens,
passwords passed as arguments). Every file agentsh itself creates to
persist that history is opened with an explicit ``0o600`` mode so it is
never left world-readable under a typical (022) process umask, and any
pre-existing file from an older, unhardened agentsh version is
re-hardened on next use.

This module intentionally does not touch history files that belong to
another program's own convention (bash's ``$HISTFILE``, PowerShell's
PSReadLine ``ConsoleHost_history.txt``) — those are only ever written
to as an explicit opt-in mirror, kept separate from this hardening.
"""

import os
from pathlib import Path

HISTORY_FILE_MODE = 0o600

_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})


def env_flag_enabled(name: str) -> bool:
    """Return True if the named environment variable is set truthy."""
    return os.environ.get(name, "").strip().lower() in _TRUE_VALUES


def _open_secure(path: Path) -> int:
    """Open path for append, creating it with HISTORY_FILE_MODE if needed.

    Also re-hardens the mode of a pre-existing file, so files created
    by an older, unhardened version of agentsh get fixed on next write.

    Returns:
        int: A raw file descriptor opened O_APPEND | O_WRONLY.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(
        path, os.O_CREAT | os.O_APPEND | os.O_WRONLY, HISTORY_FILE_MODE
    )
    try:
        os.fchmod(fd, HISTORY_FILE_MODE)
    except OSError:
        os.close(fd)
        raise
    return fd


def append_secure_line(path: Path, line: str) -> None:
    """Append line plus a trailing newline to path, hardened on write."""
    fd = _open_secure(path)
    with os.fdopen(fd, "a") as f:
        f.write(line + "\n")


def ensure_secure_file(path: Path) -> None:
    """Ensure path exists with HISTORY_FILE_MODE, without writing to it.

    Used to pre-create files that a third-party library (e.g.
    prompt_toolkit's ``FileHistory``) will later open with a bare
    ``open()`` call, so that library's create-on-first-write never
    runs under a permissive umask.
    """
    os.close(_open_secure(path))
