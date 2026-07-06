"""ContextBuilder — runs all providers concurrently with timeouts."""

import asyncio

from agentsh.context.protocol import ContextProvider
from agentsh.models import ContextFragment
from agentsh.shell.protocol import Shell


class ContextBuilder:
    """Collects context fragments from all configured providers in parallel."""

    def __init__(
        self, providers: list[ContextProvider], timeout_ms: int = 200
    ) -> None:
        """Initialise with a list of providers and a per-provider timeout."""
        self._providers = providers
        self._timeout = timeout_ms / 1000

    @property
    def provider_count(self) -> int:
        """Return the number of registered context providers."""
        return len(self._providers)

    async def build(self, shell: Shell) -> list[ContextFragment]:
        """Collect fragments; providers that failare silently dropped."""

        async def _safe_collect(p: ContextProvider) -> ContextFragment | None:
            try:
                return await asyncio.wait_for(
                    p.collect(shell), timeout=self._timeout
                )
            except Exception:
                return None

        results = await asyncio.gather(
            *(_safe_collect(p) for p in self._providers)
        )
        return [r for r in results if r is not None]
