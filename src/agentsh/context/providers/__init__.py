"""Context provider implementations, resolved dynamically by name.

Adding a provider means adding a module here: the module
``agentsh.context.providers.<name>`` must define a class named
``<Name>Provider``, mirroring how Agent.from_provider resolves
LLM backends.
"""

from collections.abc import Callable
from importlib import import_module
from typing import cast

from agentsh.context.protocol import ContextProvider

__all__ = ["UnknownProviderError", "build_providers", "resolve_provider"]


class UnknownProviderError(Exception):
    """Raised when a configured provider name cannot be resolved."""


def resolve_provider(name: str) -> Callable[[], ContextProvider]:
    """Resolve a provider factory from its module name."""
    try:
        module = import_module(f"agentsh.context.providers.{name.lower()}")
        provider_cls = getattr(module, f"{name.title()}Provider")
    except (ModuleNotFoundError, AttributeError):
        raise UnknownProviderError(
            f"Unknown context provider: {name!r}"
        ) from None
    return cast("Callable[[], ContextProvider]", provider_cls)


def build_providers(names: list[str]) -> list[ContextProvider]:
    """Instantiate the configured providers in order."""
    return [resolve_provider(name)() for name in names]
