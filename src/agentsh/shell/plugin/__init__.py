"""Shell plugins."""

from importlib import import_module
from pathlib import Path

__all__ = ["register_plugins"]


def register_plugins() -> None:
    """Dynamically import plugins to trigger registration."""
    self_file = Path(__file__)

    for module in self_file.parent.iterdir():
        if module == self_file:
            continue
        mod = module.stem
        import_module(f".{mod}", package="agentsh.shell.plugin")
