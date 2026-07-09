"""Filesystem context provider — reports cwd contents."""

import asyncio
from pathlib import Path

from agentsh.models import ContextFragment
from agentsh.shell.protocol import Shell

_MAX_FILES = 50


def _list_entries(cwd: str) -> list[str]:
    """Blocking directory listing + sort, run off the event loop."""
    entries = sorted(Path(cwd).iterdir(), key=lambda p: (p.is_file(), p.name))
    return [p.name + ("/" if p.is_dir() else "") for p in entries[:_MAX_FILES]]


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
