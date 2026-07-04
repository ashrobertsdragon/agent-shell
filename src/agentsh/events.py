"""EventBus and core event types for cross-cutting observability."""

from __future__ import annotations

import inspect
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


class EventBus:
    """Simple async event bus; subscriber exceptions are swallowed."""

    def __init__(self) -> None:
        """Initialize with an empty subscriber registry."""
        self._subscribers: dict[type, list[Callable[..., Any]]] = defaultdict(list)

    def subscribe(self, event_type: type, handler: Callable[..., Any]) -> None:
        """Register handler to be called for every published event of event_type."""
        self._subscribers[event_type].append(handler)

    async def publish(self, event: Any) -> None:
        """Deliver event to all registered subscribers; swallow handler errors."""
        for handler in self._subscribers[type(event)]:
            try:
                result = handler(event)
                if inspect.isawaitable(result):
                    await result
            except Exception:
                pass


@dataclass(frozen=True)
class CommandStarted:
    """Published immediately before a shell command is sent to the backend."""

    command: str
    cwd: str = ""


@dataclass(frozen=True)
class CommandFinished:
    """Published after a shell command returns."""

    command: str
    exit_code: int
    duration_ms: float


@dataclass(frozen=True)
class ToolInvoked:
    """Published after a tool call completes (success or error)."""

    tool_name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    success: bool = True


@dataclass(frozen=True)
class ToolDenied:
    """Published when a tool call is blocked by the permission engine."""

    tool_name: str
    key: str


@dataclass(frozen=True)
class AgentResponded:
    """Published each time the agent returns a message in the agentic loop."""

    content: str
    tool_call_count: int


@dataclass(frozen=True)
class ContextCollected:
    """Published after ContextBuilder finishes collecting all fragments."""

    provider_count: int
    fragment_count: int
