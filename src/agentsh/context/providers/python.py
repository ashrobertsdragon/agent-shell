"""Python environment context provider."""

from pathlib import Path

from agentsh.models import ContextFragment
from agentsh.shell.protocol import Shell


class PythonProvider:
    """Collects Python version and virtualenv status."""

    name = "python"

    async def collect(self, shell: Shell) -> ContextFragment | None:
        """Return Python version and venv path, or None if Python is absent.

        No stderr redirection is used here: ``CommandResult`` already
        separates stdout/stderr/exit_code regardless of what the command
        does with fd 2, and POSIX-only redirection syntax such as
        ``2>/dev/null`` breaks on cmd.exe and PowerShell. ``python3`` is
        tried first (the common POSIX convention) and ``python`` is used
        as a fallback, since Windows installs typically only provide the
        latter.
        """
        version_result = await shell.execute("python3 --version")
        if version_result.exit_code != 0 or not version_result.stdout.strip():
            version_result = await shell.execute("python --version")
        if version_result.exit_code != 0 or not version_result.stdout.strip():
            return None

        python_version = version_result.stdout.strip().removeprefix("Python ")
        cwd = shell.cwd
        has_venv = (Path(cwd) / ".venv" / "bin" / "python").is_file()

        return ContextFragment(
            provider=self.name,
            summary=f"python {python_version}",
            payload={"python_version": python_version, "has_venv": has_venv},
        )
