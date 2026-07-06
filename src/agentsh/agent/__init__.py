"""Agent layer — LLM backends and routing."""

from agentsh.agent.base import Agent

__all__ = ["Agent"]


SYSTEM_PREFIX = (
    "You are an AI assistant integrated into the user's shell. "
    "Use the provided tools to help with tasks. "
    "Be concise — you are running inside a terminal."
)
