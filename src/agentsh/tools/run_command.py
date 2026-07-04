"""RunCommand tool — executes arbitrary shell commands."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agentsh.models import CommandResult
from agentsh.shell.protocol import Shell

if TYPE_CHECKING:
    from agentsh.permissions import PermissionEngine


class RunCommand:
    """Executes a shell command through the Shell backend.

    When permissions is None all commands are allowed; the PermissionEngine
    is wired in during Task 8.
    """

    name = "RunCommand"
    description = (
        "Execute a shell command and return its stdout, stderr, and exit code."
    )
    schema: dict[str, Any] = {
        "name": "RunCommand",
        "description": (
            "Execute a shell command and return its stdout, stderr, and exit code."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute verbatim.",
                }
            },
            "required": ["command"],
        },
    }

    def __init__(self, shell: Shell, permissions: PermissionEngine | None) -> None:
        """Initialise with a Shell backend and an optional PermissionEngine."""
        self._shell = shell
        self._permissions = permissions

    async def invoke(self, **kwargs: Any) -> CommandResult:
        """Execute the given command string through the shell."""
        command: str = kwargs["command"]
        return await self._shell.execute(command)
