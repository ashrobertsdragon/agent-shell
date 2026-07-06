"""Filesystem context provider — reports cwd contents."""

from pathlib import Path

from agentsh.models import ContextFragment
from agentsh.shell.protocol import Shell

_MAX_FILES = 50


class FilesystemProvider:
    """Collects a listing of the current working directory."""

    name = "filesystem"

    async def collect(self, shell: Shell) -> ContextFragment | None:
        """Return the top-level file listing of the current directory."""
        cwd = shell.cwd
        try:
            entries = sorted(
                Path(cwd).iterdir(), key=lambda p: (p.is_file(), p.name)
            )
            files = [
                p.name + ("/" if p.is_dir() else "")
                for p in entries[:_MAX_FILES]
            ]
        except OSError:
            return None

        return ContextFragment(
            provider=self.name,
            summary=f"cwd: {cwd} ({len(files)} entries)",
            payload={"cwd": cwd, "files": files},
        )
