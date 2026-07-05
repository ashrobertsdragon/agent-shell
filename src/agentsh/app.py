"""App — top-level wiring object; holds all runtime dependencies."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from agentsh.events import EventBus
from agentsh.models import Message
from agentsh.shell.protocol import Shell
from agentsh.tools.protocol import ToolRegistry

if TYPE_CHECKING:
    from agentsh.agent import Agent
    from agentsh.context.builder import ContextBuilder
    from agentsh.permissions import PermissionEngine
    from agentsh.repl import UI


@dataclass
class AppState:
    """Mutable runtime state shared across REPL turns."""

    conversation: list[Message] = field(default_factory=list)
    max_history: int = 10

    def prune(self) -> None:
        """Trim to the last max_history messages, starting on a user turn."""
        if len(self.conversation) <= self.max_history:
            return
        trimmed = self.conversation[-self.max_history :]
        for i, msg in enumerate(trimmed):
            if msg.role == "user":
                self.conversation = list(trimmed[i:])
                return
        self.conversation = list(trimmed)


@dataclass
class App:
    """Dependency container; constructed in main.py and passed to run_repl."""

    shell: Shell
    tools: ToolRegistry
    permissions: PermissionEngine
    context_builder: ContextBuilder
    agent: Agent
    state: AppState
    event_bus: EventBus = field(default_factory=EventBus)
    ui: UI | None = None
