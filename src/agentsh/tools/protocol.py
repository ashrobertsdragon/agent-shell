"""Tool protocol and ToolRegistry."""

from typing import Protocol, TypedDict, runtime_checkable

from agentsh.models import JsonValue


class InputSchema(TypedDict):
    """TypedDict for Tool input schemas."""

    type: str
    properties: dict[str, dict[str, str]]
    required: list[str]


class SchemaDict(TypedDict):
    """TypedDict for Tool Schemas."""

    name: str
    description: str
    input_schema: InputSchema


@runtime_checkable
class Tool(Protocol):
    """Interface for an agent-callable tool."""

    name: str
    description: str
    schema: SchemaDict

    async def invoke(self, **kwargs: JsonValue) -> object:
        """Execute the tool with the given arguments."""
        ...


class ToolRegistry:
    """Registry mapping tool names to Tool instances."""

    def __init__(self) -> None:
        """Initialise with an empty registry."""
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool, overwriting existing entries with the same name."""
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        """Return a tool by name; raises KeyError if not found."""
        return self._tools[name]

    def schemas(self) -> list[SchemaDict]:
        """Return the JSON schema for every registered tool."""
        return [t.schema for t in self._tools.values()]
