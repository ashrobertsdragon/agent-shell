"""Python environment context provider."""

from __future__ import annotations

from pathlib import Path

from agentsh.models import ContextFragment
from agentsh.shell.protocol import Shell


class PythonEnvProvider:
    """Collects Python version and virtualenv status."""

    name = "python_env"

    async def collect(self, shell: Shell) -> ContextFragment | None:
        """Return Python version and venv path, or None if Python is absent."""
        version_result = await shell.execute("python3 --version 2>/dev/null")
        if version_result.exit_code != 0 or not version_result.stdout.strip():
            return None

        python_version = version_result.stdout.strip().removeprefix("Python ")
        cwd = await shell.cwd()
        has_venv = (Path(cwd) / ".venv" / "bin" / "python").is_file()

        return ContextFragment(
            provider=self.name,
            summary=f"python {python_version}",
            payload={"python_version": python_version, "has_venv": has_venv},
        )
