"""RunCommand tool — executes arbitrary shell commands with permission gating."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agentsh.models import CommandResult
from agentsh.shell.protocol import Shell

if TYPE_CHECKING:
    from agentsh.permissions import PermissionEngine, PermissionLevel


class PermissionDeniedError(Exception):
    """Raised when a command is blocked by a DENY permission rule."""


class RunCommand:
    """Executes a shell command through the Shell backend.

    When a PermissionEngine is provided:
    - DENY: raises PermissionDeniedError immediately.
    - CONFIRM: the caller (REPL or agentic loop) must prompt before calling invoke().
    - ALLOW: passes through without prompting.
    When permissions is None, all commands are allowed.
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

    def _check_key(self, command: str) -> PermissionLevel | None:
        """Return the permission level for the command, or None if no engine."""
        if self._permissions is None:
            return None

        return self._permissions.evaluate(f"RunCommand:{command}")

    async def invoke(self, **kwargs: Any) -> CommandResult:
        """Execute the given command, raising PermissionDeniedError if denied."""
        from agentsh.permissions import PermissionLevel

        command: str = kwargs["command"]
        level = self._check_key(command)
        if level == PermissionLevel.DENY:
            raise PermissionDeniedError(f"Command denied by policy: {command}")
        return await self._shell.execute(command)
