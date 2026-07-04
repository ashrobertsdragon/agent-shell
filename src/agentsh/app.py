"""App — top-level wiring object; holds all runtime dependencies."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from agentsh.models import Message
from agentsh.shell.protocol import Shell
from agentsh.tools.protocol import ToolRegistry

if TYPE_CHECKING:
    from agentsh.agent.router import AgentRouter
    from agentsh.context.builder import ContextBuilder
    from agentsh.permissions import PermissionEngine
    from agentsh.repl import UI


@dataclass
class AppState:
    """Mutable runtime state shared across REPL turns."""

    conversation: list[Message] = field(default_factory=list)


@dataclass
class App:
    """Dependency container; constructed in main.py and passed to run_repl."""

    shell: Shell
    tools: ToolRegistry
    permissions: PermissionEngine
    context_builder: ContextBuilder
    agent_router: AgentRouter
    state: AppState
    ui: UI | None = None
