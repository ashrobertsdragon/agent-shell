"""ReadFile tool — reads a file from the filesystem."""

from pathlib import Path

from agentsh.models import JsonValue
from agentsh.tools import SchemaDict


class ReadFile:
    """Reads a file and returns its contents as a string."""

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

    async def invoke(self, **kwargs: JsonValue) -> str:
        """Return the file's contents; raises FileNotFoundError if absent."""
        path = Path(str(kwargs["path"]))
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        return path.read_text(errors="replace")
