"""Determine which shell invoked the application."""

import os

_POSIX = "SHELL"
_PS = "PSModulePath"
_CMD = "CMDCMDLINE"

_SHELLS = [(_POSIX, "posix"), (_PS, "powershell"), (_CMD, "cmd")]


def detect_shell() -> str:
    """Detect the shell based on environment variables.

    Returns:
        str: The shell name.
    """
    for var, shell in _SHELLS:
        if env_var := os.environ.get(var):
            return env_var.split("/")[-1] if shell == "posix" else shell
    raise RuntimeError("Could not determine shell")
