"""ContextProvider protocol definition."""

from __future__ import annotations

from typing import Protocol

from agentsh.models import ContextFragment
from agentsh.shell.protocol import Shell


class ContextProvider(Protocol):
    """Collects a single environmental ContextFragment from the shell."""

    name: str

    async def collect(self, shell: Shell) -> ContextFragment | None:
        """Return a fragment, or None if not applicable in the current environment."""
        ...
