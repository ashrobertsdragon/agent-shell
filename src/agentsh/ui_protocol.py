"""The UI protocol shared by app.py and agent_loop.py.

`repl.py` imports `App` (to build and run the REPL), so anything that
needs to reference the concrete `repl.UI` class from `app.py` or
`agent_loop.py` would form an import cycle. Both modules only ever need
the shape of a UI, not repl's concrete implementation, so that shape is
defined here instead -- in a leaf module with no dependents of its own --
and `repl.UI` satisfies it structurally without either side importing
the other.
"""

from collections.abc import Mapping
from typing import Protocol, runtime_checkable

from agentsh.models import CommandResult, JsonValue, Message


@runtime_checkable
class UI(Protocol):
    """Interface for user-facing I/O: rendering results and confirmations."""

    def render(self, result: CommandResult | Message) -> None:
        """Print a result to the user."""
        ...

    async def confirm(
        self, tool_name: str, arguments: Mapping[str, JsonValue]
    ) -> bool:
        """Prompt the user to allow or deny a CONFIRM-level tool call."""
        ...
