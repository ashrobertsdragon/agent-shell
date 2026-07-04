"""Agent protocol definition."""

from __future__ import annotations

from typing import Any, Protocol

from agentsh.models import ContextFragment, Message


class Agent(Protocol):
    """Interface for an LLM backend."""

    async def respond(
        self,
        conversation: list[Message],
        context: list[ContextFragment],
        tools: list[dict[str, Any]],
    ) -> Message:
        """Return the next assistant message given conversation history and context."""
        ...
