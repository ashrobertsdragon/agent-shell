"""AgentRouter — selects the active agent from the configured default."""

from __future__ import annotations

from collections.abc import Mapping

from agentsh.agent.protocol import Agent
from agentsh.config import AgentConfig


class AgentRouter:
    """Routes agent requests to the configured default backend."""

    def __init__(self, config: AgentConfig, agents: Mapping[str, Agent]) -> None:
        """Initialise with agent config and a mapping of name → Agent."""
        self._config = config
        self._agents = agents

    def current(self) -> Agent:
        """Return the active agent (always the configured default for now)."""
        return self._agents[self._config.default]
