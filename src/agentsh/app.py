"""App — top-level wiring object; holds all runtime dependencies."""

from __future__ import annotations

from dataclasses import dataclass, field

from agentsh.models import Message
from agentsh.shell.protocol import Shell
from agentsh.tools.protocol import ToolRegistry


@dataclass
class AppState:
    """Mutable runtime state shared across REPL turns."""

    conversation: list[Message] = field(default_factory=list)


@dataclass
class App:
    """Dependency container; constructed in main.py and passed to run_repl."""

    shell: Shell
    tools: ToolRegistry
    state: AppState
