"""Filesystem context provider — reports cwd contents."""

import asyncio
import os

from agentsh.models import ContextFragment
from agentsh.shell.protocol import Shell

_MAX_FILES = 50


def _list_entries(cwd: str) -> list[str]:
    """Blocking directory listing + sort, run off the event loop.

    Uses os.scandir() rather than Path.iterdir() so each entry's file
    type comes from the directory read itself (DirEntry caches it) instead
    of a separate stat() syscall per entry via is_file()/is_dir().
    """
    with os.scandir(cwd) as it:
        entries = sorted(it, key=lambda e: (e.is_file(), e.name))
    return [e.name + ("/" if e.is_dir() else "") for e in entries[:_MAX_FILES]]


class FilesystemProvider:
    """Collects a listing of the current working directory."""

    name = "filesystem"

    async def collect(self, shell: Shell) -> ContextFragment | None:
        """Return the top-level file listing of the current directory."""
        cwd = shell.cwd
        try:
            files = await asyncio.to_thread(_list_entries, cwd)
        except OSError:
            return None

        return ContextFragment(
            provider=self.name,
            summary=f"cwd: {cwd} ({len(files)} entries)",
            payload={"cwd": cwd, "files": files},
        )
