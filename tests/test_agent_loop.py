"""Tests for run_agent_loop covering permission paths and iteration limit."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from agentsh.agent_loop import AgentLoopLimitError, run_agent_loop
from agentsh.events import EventBus
from agentsh.models import CommandResult, Message, ToolCall
from agentsh.permissions import PermissionLevel


def _text(content: str = "Done.") -> Message:
    """Build a terminal assistant message with no tool calls."""
    return Message(role="assistant", content=content)


def _tool_call_msg(
    tool_name: str = "RunCommand",
    args: dict[str, str] | None = None,
    call_id: str = "tc1",
) -> Message:
    """Build an assistant message containing a single tool call."""
    return Message(
        role="assistant",
        content="",
        tool_calls=(
            ToolCall(
                tool_name=tool_name,
                arguments=args if args is not None else {"command": "ls"},
                call_id=call_id,
            ),
        ),
    )


def _registry(invoke_result: object = "ok") -> MagicMock:
    """Build a ToolRegistry mock whose single tool returns invoke_result."""
    registry = MagicMock()
    registry.schemas.return_value = []
    tool = AsyncMock()
    tool.invoke.return_value = invoke_result
    registry.get.return_value = tool
    return registry


@pytest.fixture
def ui() -> MagicMock:
    """UI mock that auto-confirms all prompts."""
    m = MagicMock()
    m.confirm = AsyncMock(return_value=True)
    return m


@pytest.fixture
def allow_perms() -> MagicMock:
    """PermissionEngine mock that ALLOWs every key."""
    perms = MagicMock()
    perms.evaluate.return_value = PermissionLevel.ALLOW
    return perms


@pytest.fixture
def bus() -> EventBus:
    """Real EventBus with no subscribers."""
    return EventBus()


async def test_text_only_returns_immediately(
    ui: MagicMock,
    allow_perms: MagicMock,
    bus: EventBus,
) -> None:
    """Loop returns on the first turn when the agent produces no tool calls."""
    agent = AsyncMock()
    agent.respond.return_value = _text("All done.")
    conversation: list[Message] = [Message(role="user", content="hello")]

    result = await run_agent_loop(
        agent=agent,
        conversation=conversation,
        context=[],
        tools=_registry(),
        permissions=allow_perms,
        ui=ui,
        event_bus=bus,
    )

    assert result.content == "All done."
    agent.respond.assert_called_once()


async def test_tool_call_executes_and_loop_continues(
    ui: MagicMock,
    allow_perms: MagicMock,
    bus: EventBus,
) -> None:
    """Tool is invoked and the follow-up response is returned."""
    agent = AsyncMock()
    agent.respond.side_effect = [_tool_call_msg(), _text("Command ran.")]

    cmd_result = CommandResult(
        stdout="file.txt\n", stderr="", exit_code=0, duration_ms=5.0, cwd="/tmp"
    )
    registry = _registry(invoke_result=cmd_result)
    conversation: list[Message] = [Message(role="user", content="list files")]

    result = await run_agent_loop(
        agent=agent,
        conversation=conversation,
        context=[],
        tools=registry,
        permissions=allow_perms,
        ui=ui,
        event_bus=bus,
    )

    assert result.content == "Command ran."
    assert agent.respond.call_count == 2
    registry.get.return_value.invoke.assert_called_once_with(command="ls")


async def test_deny_injects_error_tool_result(
    ui: MagicMock,
    bus: EventBus,
) -> None:
    """A DENY-blocked call injects an error ToolResult so the agent can recover."""
    perms = MagicMock()
    perms.evaluate.return_value = PermissionLevel.DENY

    agent = AsyncMock()
    agent.respond.side_effect = [_tool_call_msg(), _text("Blocked.")]

    registry = MagicMock()
    registry.schemas.return_value = []

    conversation: list[Message] = [Message(role="user", content="run cmd")]
    await run_agent_loop(
        agent=agent,
        conversation=conversation,
        context=[],
        tools=registry,
        permissions=perms,
        ui=ui,
        event_bus=bus,
    )

    tool_msg = conversation[-2]
    assert tool_msg.role == "tool"
    assert tool_msg.tool_results[0].is_error is True
    assert "denied" in tool_msg.tool_results[0].content.lower()


async def test_confirm_denied_by_user_injects_error_tool_result(
    bus: EventBus,
) -> None:
    """When the user declines a CONFIRM prompt, an error ToolResult is injected."""
    perms = MagicMock()
    perms.evaluate.return_value = PermissionLevel.CONFIRM

    ui = MagicMock()
    ui.confirm = AsyncMock(return_value=False)

    agent = AsyncMock()
    agent.respond.side_effect = [_tool_call_msg(), _text("User declined.")]

    registry = MagicMock()
    registry.schemas.return_value = []

    conversation: list[Message] = [Message(role="user", content="do something")]
    await run_agent_loop(
        agent=agent,
        conversation=conversation,
        context=[],
        tools=registry,
        permissions=perms,
        ui=ui,
        event_bus=bus,
    )

    tool_msg = conversation[-2]
    assert tool_msg.tool_results[0].is_error is True
    assert "user" in tool_msg.tool_results[0].content.lower()


async def test_iteration_limit_raises(
    ui: MagicMock,
    allow_perms: MagicMock,
    bus: EventBus,
) -> None:
    """AgentLoopLimitError is raised after max_iterations with no terminal response."""
    agent = AsyncMock()
    agent.respond.return_value = _tool_call_msg()

    conversation: list[Message] = [Message(role="user", content="loop forever")]

    with pytest.raises(AgentLoopLimitError):
        await run_agent_loop(
            agent=agent,
            conversation=conversation,
            context=[],
            tools=_registry(),
            permissions=allow_perms,
            ui=ui,
            event_bus=bus,
            max_iterations=3,
        )

    assert agent.respond.call_count == 3
