"""Tool protocol and ToolRegistry."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Tool(Protocol):
    """Interface for an agent-callable tool."""

    name: str
    description: str
    schema: dict[str, Any]

    async def invoke(self, **kwargs: Any) -> Any:
        """Execute the tool with the given arguments."""
        ...


class ToolRegistry:
    """Registry mapping tool names to Tool instances."""

    def __init__(self) -> None:
        """Initialise with an empty registry."""
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool, overwriting any existing entry with the same name."""
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        """Return a tool by name; raises KeyError if not found."""
        return self._tools[name]

    def schemas(self) -> list[dict[str, Any]]:
        """Return the JSON schema for every registered tool."""
        return [t.schema for t in self._tools.values()]
