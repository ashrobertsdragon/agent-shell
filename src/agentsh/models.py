"""Core data models shared across all agentsh layers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class CommandResult:
    """Result of a shell command execution."""

    stdout: str
    stderr: str
    exit_code: int
    duration_ms: float
    cwd: str


@dataclass(frozen=True, slots=True)
class ToolCall:
    """A request from the LLM to invoke a tool."""

    tool_name: str
    arguments: dict[str, Any]
    call_id: str


@dataclass(frozen=True, slots=True)
class ToolResult:
    """The outcome of a tool invocation."""

    call_id: str
    content: str
    is_error: bool = False


@dataclass(frozen=True, slots=True)
class Message:
    """A single message in the LLM conversation history."""

    role: str
    content: str
    tool_calls: tuple[ToolCall, ...] = field(default_factory=tuple)
    tool_results: tuple[ToolResult, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class ContextFragment:
    """A piece of context collected from a context provider."""

    provider: str
    summary: str
    payload: dict[str, Any]
