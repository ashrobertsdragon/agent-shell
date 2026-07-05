"""Agent protocol definition."""

import importlib
from typing import TYPE_CHECKING, cast

from agentsh.models import ContextFragment, Message

if TYPE_CHECKING:
    from agentsh.config import AgentConfig
    from agentsh.tools import SchemaDict


class Agent:
    """Interface for an LLM backend."""

    @classmethod
    def from_provider(cls, provider: str) -> type["Agent"]:
        """Resolve an Agent subclass from a provider name."""
        module = importlib.import_module(f"agentsh.agent.{provider.lower()}")
        agent_cls = getattr(module, f"{provider.title()}Agent")
        return cast("type[Agent]", agent_cls)

    def __init__(self, config: AgentConfig) -> None:
        """Initialise the async Anthropic client."""
        raise NotImplementedError

    async def respond(
        self,
        conversation: list[Message],
        context: list[ContextFragment],
        tools: list[SchemaDict],
    ) -> Message:
        """Return the next assistant message."""
        raise NotImplementedError
