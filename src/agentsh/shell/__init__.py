"""Shell Plugins."""

from agentsh.shell._detect import detect_shell
from agentsh.shell._registry import available, get
from agentsh.shell.plugin import register_plugins
from agentsh.shell.protocol import Shell

__all__ = [
    "Shell",
    "UnsupportedShellError",
    "available",
    "create_shell",
    "get",
]

register_plugins()

_CONFIG_HINT = (
    "Set shell explicitly in ~/.config/agentsh/config.toml to override."
)


class UnsupportedShellError(Exception):
    """Raised when the configured or detected shell has no backend."""


def create_shell(shell_name: str) -> Shell:
    """Initialize a shell plugin from config value.

    Raises:
        UnsupportedShellError: If the shell cannot be detected or no
            backend is registered for it.
    """
    try:
        name = detect_shell() if shell_name == "auto" else shell_name
    except RuntimeError as e:
        raise UnsupportedShellError(f"{e}. {_CONFIG_HINT}") from None
    try:
        shell_cls = get(name)
    except KeyError:
        supported = ", ".join(available())
        raise UnsupportedShellError(
            f"Shell {name!r} is not supported yet."
            f" Supported shells: {supported}. {_CONFIG_HINT}"
        ) from None
    return shell_cls()
