"""Tests for the RunCommand tool."""

from unittest.mock import AsyncMock

import pytest

from agentsh.config import PermissionRulesConfig
from agentsh.models import CommandResult
from agentsh.permissions import PermissionEngine
from agentsh.tools.protocol import ToolRegistry
from agentsh.tools.run_command import PermissionDeniedError, RunCommand


@pytest.fixture
def mock_shell() -> AsyncMock:
    """Shell mock that returns a fixed CommandResult."""
    shell = AsyncMock()
    shell.execute = AsyncMock(
        return_value=CommandResult(
            stdout="hello\n", stderr="", exit_code=0, duration_ms=5.0, cwd="/tmp"
        )
    )
    return shell


async def test_run_command_invokes_shell(mock_shell: AsyncMock) -> None:
    """invoke delegates to shell.execute and returns its result."""
    tool = RunCommand(shell=mock_shell, permissions=None)
    result = await tool.invoke(command="echo hello")
    mock_shell.execute.assert_called_once_with("echo hello")
    assert result.stdout == "hello\n"


async def test_run_command_deny_raises(mock_shell: AsyncMock) -> None:
    """A DENY-matched command raises PermissionDeniedError."""
    rules = PermissionRulesConfig(deny=("RunCommand:rm*",))
    permissions = PermissionEngine(rules)
    tool = RunCommand(shell=mock_shell, permissions=permissions)
    with pytest.raises(PermissionDeniedError):
        await tool.invoke(command="rm -rf /")


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
