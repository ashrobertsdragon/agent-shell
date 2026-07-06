"""Shell plugins."""

from importlib import import_module
from pathlib import Path

__all__ = ["register_plugins"]


def register_plugins() -> None:
    """Dynamically import plugin modules to trigger registration."""
    package_dir = Path(__file__).parent

    for module in package_dir.glob("*.py"):
        if module.name.startswith("_"):
            continue
        import_module(f".{module.stem}", package="agentsh.shell.plugin")
