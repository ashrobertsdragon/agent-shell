"""Off-event-loop helpers for the per-command stderr scratch file.

Every persistent shell backend redirects a command's stderr to a fresh
temp file, reads it back once the command completes, then deletes it.
Creating, reading, and deleting that file are all blocking filesystem
calls; callers must run each of them via ``asyncio.to_thread`` rather
than inline on the event loop, or a single slow command stalls every
other coroutine waiting on the loop.
"""

import os
import tempfile
from pathlib import Path


def create_stderr_tempfile(prefix: str = "agentsh_stderr_") -> str:
    """Create an empty temp file to capture a command's redirected stderr."""
    fd, path = tempfile.mkstemp(prefix=prefix)
    os.close(fd)
    return path


def discard_stderr_tempfile(path: str) -> None:
    """Remove a stderr temp file, ignoring it if already gone."""
    Path(path).unlink(missing_ok=True)
