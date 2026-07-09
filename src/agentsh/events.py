"""EventBus and core event types for cross-cutting observability."""

import inspect
import logging
from collections import defaultdict
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TypeVar, cast

from agentsh.models import JsonValue

E = TypeVar("E")

logger = logging.getLogger(__name__)


class EventBus:
    """Simple async event bus.

    A subscriber's exception is caught and logged (with traceback)
    rather than propagated, so one broken subscriber cannot stop
    delivery to the rest or crash the publish loop — but the failure
    is observable instead of silently discarded.
    """

    def __init__(self) -> None:
        """Initialize with an empty subscriber registry."""
        self._subscribers: dict[
            type[object], list[Callable[[object], object]]
        ] = defaultdict(list)

    def subscribe(
        self, event_type: type[E], handler: Callable[[E], object]
    ) -> None:
        """Register handler to be called for published event of event_type."""
        self._subscribers[event_type].append(
            cast(Callable[[object], object], handler)
        )

    async def publish(self, event: object) -> None:
        """Deliver event to all subscribers, logging any handler errors."""
        for handler in self._subscribers[type(event)]:
            try:
                result = handler(event)
                if inspect.isawaitable(result):
                    await result
            except Exception:
                logger.exception(
                    "Unhandled exception in event subscriber %r for %s",
                    handler,
                    type(event).__name__,
                )


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
    arguments: Mapping[str, JsonValue] = field(default_factory=dict)
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
