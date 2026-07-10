"""Agent protocol definition."""

from importlib import import_module
from typing import TYPE_CHECKING

from agentsh.models import ContextFragment, Message
from agentsh.registry import Registry

if TYPE_CHECKING:
    from agentsh.config import AgentConfig
    from agentsh.tools import SchemaDict


class Agent:
    """Interface for an LLM backend."""

    @classmethod
    def from_provider(cls, provider: str) -> type["Agent"]:
        """Resolve the Agent subclass registered for provider.

        Importing ``agentsh.agent.<provider>`` triggers that module's
        ``@register(name)`` decorator (see agentsh.registry.Registry) as
        a side effect, so resolution never depends on guessing a class
        name from `provider`. Only the requested backend module is
        imported -- not every backend eagerly -- since each backend
        depends on an optional, per-provider third-party SDK (anthropic,
        openai, google-genai, openrouter) that may not be installed.
        """
        import_module(f"agentsh.agent.{provider.lower()}")
        return _registry.get(provider)

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


_registry: Registry[Agent] = Registry()

register = _registry.register
