"""WriteFile tool — writes or patches a file on the filesystem."""

import re
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from agentsh.models import JsonValue
from agentsh.permissions import (
    ConfirmCallback,
    PermissionDeniedError,
    PermissionEngine,
)
from agentsh.tools import SchemaDict
from agentsh.tools._paths import canonical_path, is_within_roots

_BLOCK_RE = re.compile(
    r"<<<<<<< SEARCH\n(.*?)\n=======\n(.*?)\n>>>>>>> REPLACE",
    re.DOTALL,
)


def _apply_patch(original: str, patch: str) -> str:
    """Apply SEARCH/REPLACE blocks from patch to original, in order."""
    result = original
    blocks = _BLOCK_RE.findall(patch)
    if not blocks:
        raise ValueError("Patch contains no valid SEARCH/REPLACE blocks.")
    for search, replacement in blocks:
        if search not in result:
            raise ValueError(f"Search text not found in file: {search[:80]!r}")
        result = result.replace(search, replacement, 1)
    return result


class WriteFile:
    """Writes content to a file, or applies a SEARCH/REPLACE patch.

    Every call is gated by the mandatory PermissionEngine:
    - DENY: raises PermissionDeniedError immediately.
    - CONFIRM: the injected confirm callback is awaited; raises if none
      is configured or if it declines.
    - ALLOW: passes through without prompting.
    """

    name = "WriteFile"
    description = (
        "Write content to a file (full overwrite), or apply targeted"
        " SEARCH/REPLACE edits using the patch parameter."
    )
    schema: SchemaDict = {
        "name": "WriteFile",
        "description": (
            "Write content to a file (full overwrite), or apply targeted"
            " SEARCH/REPLACE edits using the patch parameter."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute or relative path to the file.",
                },
                "content": {
                    "type": "string",
                    "description": (
                        "Full file content for a complete overwrite."
                        " Mutually exclusive with patch."
                    ),
                },
                "patch": {
                    "type": "string",
                    "description": (
                        "One or more SEARCH/REPLACE blocks. Format: "
                        "<<<<<<< SEARCH\\n"
                        "<old>\\n"
                        "=======\\n"
                        "<new>\\n"
                        ">>>>>>> REPLACE"
                    ),
                },
            },
            "required": ["path"],
        },
    }

    def __init__(
        self,
        permissions: PermissionEngine,
        confirm: ConfirmCallback | None = None,
        sandbox_roots: Sequence[Path] | None = None,
    ) -> None:
        """Initialise with a mandatory PermissionEngine.

        confirm is awaited for CONFIRM-level paths; if None, such writes
        are refused rather than silently applied.

        sandbox_roots, if given, confines every write to those directory
        trees regardless of what the PermissionEngine allows: a broad
        allow rule (or an approved CONFIRM) only clears the policy gate,
        it must not also grant the tool license to create arbitrary
        directory trees anywhere the process can write. Leaving it unset
        preserves the previous unconfined behavior.

        Each root is re-resolved via canonical_path() here rather than
        trusting the caller to have already done so, so containment
        checks stay correct even if a future caller passes a relative,
        symlinked, or ~-prefixed root.
        """
        self._permissions = permissions
        self._confirm = confirm
        self._sandbox_roots = (
            [canonical_path(str(root)) for root in sandbox_roots]
            if sandbox_roots
            else []
        )

    async def invoke(self, **kwargs: JsonValue) -> str:
        """Write or patch the file after enforcing permissions.

        Raises:
            PermissionDeniedError: if denied by policy, if CONFIRM is
                required and no confirm callback approves the call, or if
                sandbox_roots is configured and the path falls outside it.
            ValueError: if neither content nor patch is supplied, or the
                patch does not apply cleanly.
        """
        content: str | None = cast(str | None, kwargs.get("content"))
        patch: str | None = cast(str | None, kwargs.get("patch"))

        if patch is None and content is None:
            raise ValueError("WriteFile requires either content or patch.")

        path = canonical_path(str(kwargs["path"]))

        if self._sandbox_roots and not is_within_roots(
            path, self._sandbox_roots
        ):
            raise PermissionDeniedError(
                f"WriteFile denied: {path} is outside the allowed sandbox roots"
            )

        await self._permissions.enforce(
            "WriteFile", {"path": path.as_posix()}, self._confirm
        )

        path.parent.mkdir(parents=True, exist_ok=True)

        if patch is not None:
            original = (
                path.read_text(encoding="utf-8", errors="replace")
                if path.exists()
                else ""
            )
            path.write_text(_apply_patch(original, patch), encoding="utf-8")
        else:
            path.write_text(content or "", encoding="utf-8")

        return f"Written: {path}"
