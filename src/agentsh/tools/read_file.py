"""ReadFile tool — reads a file from the filesystem."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class ReadFile:
    """Reads a file and returns its contents as a string."""

    name = "ReadFile"
    description = "Read the contents of a file at the given path."
    schema: dict[str, Any] = {
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

    async def invoke(self, **kwargs: Any) -> str:
        """Return the file's contents; raises FileNotFoundError if absent."""
        path = Path(kwargs["path"])
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        return path.read_text(errors="replace")
