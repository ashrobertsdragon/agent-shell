"""RunCommand tool — executes shell commands with permission gating."""

from dataclasses import replace

from agentsh.limits import truncate_text
from agentsh.models import CommandResult, JsonValue
from agentsh.permissions import ConfirmCallback, PermissionEngine
from agentsh.shell.protocol import Shell
from agentsh.tools import SchemaDict


class RunCommand:
    """Executes a shell command through the Shell backend.

    Every call is gated by the mandatory PermissionEngine:
    - DENY: raises PermissionDeniedError immediately.
    - CONFIRM: the injected confirm callback is awaited; raises if none
      is configured or if it declines.
    - ALLOW: passes through without prompting.
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
        self,
        shell: Shell,
        permissions: PermissionEngine,
        confirm: ConfirmCallback | None = None,
    ) -> None:
        """Initialise with a Shell backend, a mandatory PermissionEngine.

        confirm is awaited for CONFIRM-level commands; if None, such
        commands are refused rather than silently executed.
        """
        self._shell = shell
        self._permissions = permissions
        self._confirm = confirm

    async def invoke(self, **kwargs: JsonValue) -> CommandResult:
        """Execute the given command after enforcing permissions.

        The shell backend already caps its own stdout/stderr, but the
        result is re-capped here too as defense-in-depth against any
        Shell implementation that does not.

        Raises:
            PermissionDeniedError: if denied by policy, or if CONFIRM is
                required and no confirm callback approves the call.
        """
        command = str(kwargs["command"])
        await self._permissions.enforce(
            "RunCommand", {"command": command}, self._confirm
        )
        result = await self._shell.execute(command)
        return replace(
            result,
            stdout=truncate_text(result.stdout),
            stderr=truncate_text(result.stderr),
        )
