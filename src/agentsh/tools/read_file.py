"""ReadFile tool — reads a file from the filesystem."""

from agentsh.models import JsonValue
from agentsh.permissions import ConfirmCallback, PermissionEngine
from agentsh.tools import SchemaDict
from agentsh.tools._paths import canonical_path


class ReadFile:
    """Reads a file and returns its contents as a string.

    Every call is gated by the mandatory PermissionEngine:
    - DENY: raises PermissionDeniedError immediately.
    - CONFIRM: the injected confirm callback is awaited; raises if none
      is configured or if it declines.
    - ALLOW: passes through without prompting.
    """

    name = "ReadFile"
    description = "Read the contents of a file at the given path."
    schema: SchemaDict = {
        "name": "ReadFile",
        "description": "Read the contents of a file at the given path.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute or relative path to the file.",
                }
            },
            "required": ["path"],
        },
    }

    def __init__(
        self,
        permissions: PermissionEngine,
        confirm: ConfirmCallback | None = None,
    ) -> None:
        """Initialise with a mandatory PermissionEngine.

        confirm is awaited for CONFIRM-level paths; if None, such reads
        are refused rather than silently allowed.
        """
        self._permissions = permissions
        self._confirm = confirm

    async def invoke(self, **kwargs: JsonValue) -> str:
        """Return the file's contents after enforcing permissions.

        Raises:
            PermissionDeniedError: if denied by policy, or if CONFIRM is
                required and no confirm callback approves the call.
            FileNotFoundError: if the file does not exist.
        """
        raw_path = str(kwargs["path"])
        await self._permissions.enforce(
            "ReadFile", {"path": raw_path}, self._confirm
        )
        path = canonical_path(raw_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        return path.read_text(encoding="utf-8", errors="replace")
