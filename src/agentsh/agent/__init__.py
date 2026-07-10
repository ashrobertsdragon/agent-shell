"""Agent layer — LLM backends and routing."""

from agentsh.agent._system import SYSTEM_PREFIX, _build_system
from agentsh.agent.base import Agent

__all__ = ["Agent", "SYSTEM_PREFIX", "_build_system"]
