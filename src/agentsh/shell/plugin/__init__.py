"""Shell plugins."""

from pathlib import Path

from agentsh.registry import discover_modules

__all__ = ["register_plugins"]


def register_plugins() -> None:
    """Dynamically import plugin modules to trigger registration."""
    package_dir = Path(__file__).parent
    discover_modules(package_dir, "agentsh.shell.plugin")
