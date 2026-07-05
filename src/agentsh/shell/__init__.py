"""Shell Plugins."""

from agentsh.shell._detect import detect_shell
from agentsh.shell._registry import available, get
from agentsh.shell.plugin import register_plugins
from agentsh.shell.protocol import Shell

__all__ = ["Shell", "available", "create_shell", "get"]

register_plugins()


def create_shell(shell_name: str) -> Shell:
    """Initialize a shell plugin from config value."""
    name = detect_shell() if shell_name == "auto" else shell_name
    return get(name)()
