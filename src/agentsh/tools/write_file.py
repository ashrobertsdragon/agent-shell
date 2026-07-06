"""WriteFile tool — writes or patches a file on the filesystem."""

import re
from typing import cast

from agentsh.models import JsonValue
from agentsh.tools import SchemaDict
from agentsh.tools._paths import canonical_path

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
    """Writes content to a file, or applies a SEARCH/REPLACE patch."""

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

    async def invoke(self, **kwargs: JsonValue) -> str:
        """Write or patch the file; returns a confirmation string."""
        path = canonical_path(str(kwargs["path"]))
        content: str | None = cast(str | None, kwargs.get("content"))
        patch: str | None = cast(str | None, kwargs.get("patch"))

        if patch is None and content is None:
            raise ValueError("WriteFile requires either content or patch.")

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
