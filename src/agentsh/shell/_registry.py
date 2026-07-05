"""Shell plugin registration."""

from collections.abc import Callable
from functools import cache

from agentsh.shell.protocol import Shell

_registry: dict[str, type[Shell]] = {}


def register(name: str) -> Callable[[type[Shell]], type[Shell]]:
    """Decorator to register shell plugins.

    Args:
        name (str): The name of the shell plugin.

    Returns:
        Callable: The decorated function.
    """

    def _decorator(cls: type[Shell]) -> type[Shell]:

        _registry[name.lower()] = cls
        return cls

    return _decorator


@cache
def get(name: str) -> type[Shell]:
    """Getter for Shell plugin by name."""
    return _registry[name.lower()]


@cache
def available() -> list[str]:
    """List all registered shell plugins.

    Returns:
        list[str]: All the registered shell plugins by name.
    """
    return sorted(_registry.keys())
