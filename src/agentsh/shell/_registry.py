"""Shell plugin registration."""

from functools import cache

from agentsh.registry import Registry
from agentsh.shell.protocol import Shell

_registry: Registry[Shell] = Registry()

register = _registry.register


@cache
def get(name: str) -> type[Shell]:
    """Getter for Shell plugin by name."""
    return _registry.get(name)


@cache
def available() -> list[str]:
    """List all registered shell plugins.

    Returns:
        list[str]: All the registered shell plugins by name.
    """
    return _registry.available()
