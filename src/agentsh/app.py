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
        """Trim old turns so the conversation fits max_history.

        The cut always lands on a user message so no assistant/tool
        message is orphaned (providers reject tool results without their
        originating call). The most recent turn is never split, even if
        it alone exceeds max_history.
        """
        if len(self.conversation) <= self.max_history:
            return
        user_indices = [
            i for i, msg in enumerate(self.conversation) if msg.role == "user"
        ]
        if not user_indices:
            return
        cut = user_indices[-1]
        for idx in reversed(user_indices):
            if len(self.conversation) - idx > self.max_history:
                break
            cut = idx
        self.conversation = self.conversation[cut:]


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
