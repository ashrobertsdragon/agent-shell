"""Tests for the RunCommand tool."""

from unittest.mock import AsyncMock

import pytest

from agentsh.config import PermissionRulesConfig
from agentsh.limits import MAX_OUTPUT_BYTES, truncation_marker
from agentsh.models import CommandResult
from agentsh.permissions import PermissionDeniedError, PermissionEngine
from agentsh.tools.protocol import ToolRegistry
from agentsh.tools.run_command import RunCommand


@pytest.fixture
def mock_shell() -> AsyncMock:
    """Shell mock that returns a fixed CommandResult."""
    shell = AsyncMock()
    shell.execute = AsyncMock(
        return_value=CommandResult(
            stdout="hello\n",
            stderr="",
            exit_code=0,
            duration_ms=5.0,
            cwd="/tmp",
        )
    )
    return shell


@pytest.fixture
def allow_all() -> PermissionEngine:
    """PermissionEngine that ALLOWs every RunCommand call."""
    return PermissionEngine(PermissionRulesConfig(allow={"RunCommand:*"}))


async def test_run_command_invokes_shell(
    mock_shell: AsyncMock, allow_all: PermissionEngine
) -> None:
    """invoke delegates to shell.execute and returns its result."""
    tool = RunCommand(shell=mock_shell, permissions=allow_all)
    result = await tool.invoke(command="echo hello")
    mock_shell.execute.assert_called_once_with("echo hello")
    assert result.stdout == "hello\n"


async def test_run_command_truncates_oversized_shell_output(
    allow_all: PermissionEngine,
) -> None:
    """invoke re-caps stdout/stderr as defense-in-depth against the shell.

    The Shell backend already caps its own output, but RunCommand must
    not blindly trust every Shell implementation to do so.
    """
    shell = AsyncMock()
    shell.execute = AsyncMock(
        return_value=CommandResult(
            stdout="a" * (MAX_OUTPUT_BYTES + 4096),
            stderr="b" * (MAX_OUTPUT_BYTES + 4096),
            exit_code=0,
            duration_ms=5.0,
            cwd="/tmp",
        )
    )
    tool = RunCommand(shell=shell, permissions=allow_all)
    result = await tool.invoke(command="cat huge-file")
    assert result.stdout.endswith(truncation_marker(MAX_OUTPUT_BYTES))
    assert result.stderr.endswith(truncation_marker(MAX_OUTPUT_BYTES))
    assert len(result.stdout.encode()) <= MAX_OUTPUT_BYTES + len(
        truncation_marker(MAX_OUTPUT_BYTES).encode()
    )


async def test_run_command_deny_raises(mock_shell: AsyncMock) -> None:
    """A DENY-matched command raises PermissionDeniedError."""
    rules = PermissionRulesConfig(deny={"RunCommand:rm*"})
    permissions = PermissionEngine(rules)
    tool = RunCommand(shell=mock_shell, permissions=permissions)
    with pytest.raises(PermissionDeniedError):
        await tool.invoke(command="rm -rf /")
    mock_shell.execute.assert_not_called()


async def test_run_command_confirm_blocks_without_callback(
    mock_shell: AsyncMock,
) -> None:
    """A CONFIRM-matched command is blocked when invoked directly with no
    confirm callback wired in — this is the bypass scenario the tool's
    internal enforcement exists to close (calling the tool without going
    through the agent loop at all).
    """
    rules = PermissionRulesConfig(confirm={"RunCommand:git commit*"})
    permissions = PermissionEngine(rules)
    tool = RunCommand(shell=mock_shell, permissions=permissions)
    with pytest.raises(PermissionDeniedError):
        await tool.invoke(command="git commit -m 'x'")
    mock_shell.execute.assert_not_called()


async def test_run_command_confirm_blocks_when_callback_declines(
    mock_shell: AsyncMock,
) -> None:
    """A CONFIRM-matched command is blocked when the confirm callback
    declines the call.
    """
    rules = PermissionRulesConfig(confirm={"RunCommand:git commit*"})
    permissions = PermissionEngine(rules)
    confirm = AsyncMock(return_value=False)
    tool = RunCommand(
        shell=mock_shell, permissions=permissions, confirm=confirm
    )
    with pytest.raises(PermissionDeniedError):
        await tool.invoke(command="git commit -m 'x'")
    confirm.assert_awaited_once_with(
        "RunCommand", {"command": "git commit -m 'x'"}
    )
    mock_shell.execute.assert_not_called()


async def test_run_command_confirm_proceeds_when_callback_approves(
    mock_shell: AsyncMock,
) -> None:
    """A CONFIRM-matched command executes once the confirm callback
    approves it.
    """
    rules = PermissionRulesConfig(confirm={"RunCommand:git commit*"})
    permissions = PermissionEngine(rules)
    confirm = AsyncMock(return_value=True)
    tool = RunCommand(
        shell=mock_shell, permissions=permissions, confirm=confirm
    )
    result = await tool.invoke(command="git commit -m 'x'")
    mock_shell.execute.assert_called_once_with("git commit -m 'x'")
    assert result.stdout == "hello\n"
    confirm.assert_awaited_once_with(
        "RunCommand", {"command": "git commit -m 'x'"}
    )


async def test_run_command_allow_wildcard_does_not_bypass_metacharacters(
    mock_shell: AsyncMock,
) -> None:
    """A wildcard allow rule cannot fnmatch its way past a chained command.

    ``RunCommand:git *`` matching ``git; rm -rf /`` would be a shell
    metacharacter bypass; the engine must force CONFIRM instead, and
    without a confirm callback the tool refuses to execute it.
    """
    rules = PermissionRulesConfig(allow={"RunCommand:git *"})
    permissions = PermissionEngine(rules)
    tool = RunCommand(shell=mock_shell, permissions=permissions)
    with pytest.raises(PermissionDeniedError):
        await tool.invoke(command="git; rm -rf /")
    mock_shell.execute.assert_not_called()


def test_tool_registry_get() -> None:
    """get returns the registered tool by name."""
    registry = ToolRegistry()
    tool = AsyncMock()
    tool.name = "RunCommand"
    registry.register(tool)
    assert registry.get("RunCommand") is tool


def test_tool_registry_schemas() -> None:
    """schemas returns the schema for every registered tool."""
    registry = ToolRegistry()
    tool = AsyncMock()
    tool.name = "RunCommand"
    tool.schema = {"name": "RunCommand"}
    registry.register(tool)
    assert registry.schemas() == [{"name": "RunCommand"}]
