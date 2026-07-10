"""Context provider implementations, registered via decorator.

Adding a provider means adding a module here whose ContextProvider
class carries a ``@register("name")`` decorator (see
agentsh.registry.Registry). Resolving a provider imports
``agentsh.context.providers.<name>`` to trigger that module's
registration as a side effect, then looks the class up by name --
mirroring how agent backends are resolved in agentsh.agent.base, but
importing only the one requested module rather than discovering the
whole package eagerly (kept lazy, as before, though providers unlike
agent backends have no optional third-party dependencies).
"""

from importlib import import_module

from agentsh.context.protocol import ContextProvider
from agentsh.registry import Registry

__all__ = [
    "UnknownProviderError",
    "build_providers",
    "register",
    "resolve_provider",
]

_registry: Registry[ContextProvider] = Registry()

register = _registry.register


class UnknownProviderError(Exception):
    """Raised when a configured provider name cannot be resolved."""


def resolve_provider(name: str) -> type[ContextProvider]:
    """Resolve a provider class by name.

    Importing the provider's module runs its ``@register(name)``
    decorator, so lookup never depends on guessing a class name from
    ``name`` -- see agentsh.registry.Registry.
    """
    try:
        import_module(f"agentsh.context.providers.{name.lower()}")
    except ModuleNotFoundError:
        raise UnknownProviderError(
            f"Unknown context provider: {name!r}"
        ) from None
    try:
        return _registry.get(name)
    except KeyError:
        raise UnknownProviderError(
            f"Unknown context provider: {name!r}"
        ) from None


def build_providers(names: list[str]) -> list[ContextProvider]:
    """Instantiate the configured providers in order."""
    return [resolve_provider(name)() for name in names]
