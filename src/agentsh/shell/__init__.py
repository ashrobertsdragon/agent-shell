"""Shell abstraction layer."""

from agentsh.shell.bash import BashShell
from agentsh.shell.protocol import Shell

__all__ = ["BashShell", "Shell"]
