"""RunCommand tool — executes shell commands with permission gating."""

from agentsh.models import CommandResult, JsonValue
from agentsh.permissions import PermissionEngine, PermissionLevel
from agentsh.shell.protocol import Shell
from agentsh.tools import SchemaDict


class PermissionDeniedError(Exception):
    """Raised when a command is blocked by a DENY permission rule."""


class RunCommand:
    """Executes a shell command through the Shell backend.

    When a PermissionEngine is provided:
    - DENY: raises PermissionDeniedError immediately.
    - CONFIRM: the caller must prompt before calling invoke().
    - ALLOW: passes through without prompting.
    When permissions is None, all commands are allowed.
    """

    name = "RunCommand"
    description = (
        "Execute a shell command and return its stdout, stderr, and exit code."
    )
    schema: SchemaDict = {
        "name": "RunCommand",
        "description": (
            "Execute a shell command and return stdout, stderr, and exit code."
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

    def __init__(
        self, shell: Shell, permissions: PermissionEngine | None
    ) -> None:
        """Initialise with a Shell backend and an optional PermissionEngine."""
        self._shell = shell
        self._permissions = permissions

    def _check_key(self, command: str) -> PermissionLevel:
        """Return the permission level for the command, or None if no engine."""
        if self._permissions is None:
            return PermissionLevel.ALLOW

        return self._permissions.evaluate(f"RunCommand:{command}")

    async def invoke(self, **kwargs: JsonValue) -> CommandResult:
        """Execute the given command.

        Raises:
            PermissionDeniedError if denied
        """
        command = str(kwargs["command"])
        if self._check_key(command) == PermissionLevel.DENY:
            raise PermissionDeniedError(f"Command denied by policy: {command}")
        return await self._shell.execute(command)
