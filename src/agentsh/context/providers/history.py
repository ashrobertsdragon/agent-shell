"""History context provider — reports recent shell commands."""

from __future__ import annotations

from agentsh.models import ContextFragment
from agentsh.shell.protocol import Shell

_LIMIT = 20


class HistoryProvider:
    """Collects the most recent shell history entries."""

    name = "history"

    async def collect(self, shell: Shell) -> ContextFragment | None:
        """Return the last N history entries, or None if history is empty."""
        entries = await shell.history(limit=_LIMIT)
        if not entries:
            return None

        return ContextFragment(
            provider=self.name,
            summary=f"last {len(entries)} commands",
            payload={"recent": entries},
        )
