"""Regression tests: RunCommand.invoke() is not fooled by whitespace padding.

This file originally pinned a real defect: RunCommand built its own
permission key from the raw, unstripped command, while agent_loop's
tool_call_key stripped it first (see tests/test_permission_keys.py::
test_run_command_key_strips_whitespace), so a whitespace-padded command
could dodge a DENY glob when RunCommand.invoke() was called directly.

That defect no longer reproduces: RunCommand.invoke() now delegates to
PermissionEngine.enforce(), which builds its key via the single
tool_call_key() helper -- the same one the agent loop uses -- so both
call sites strip identically. These tests confirm that fix and guard
against a future regression reintroducing the mismatch.
"""

from unittest.mock import AsyncMock

import pytest

from agentsh.config import PermissionsConfig
from agentsh.permissions import (
    PermissionDeniedError,
    PermissionEngine,
    PermissionLevel,
)
from agentsh.tools.run_command import RunCommand


@pytest.fixture
def mock_shell() -> AsyncMock:
    """Shell mock whose execute() we can assert was (not) reached."""
    return AsyncMock()


async def test_stripped_command_is_denied(mock_shell: AsyncMock) -> None:
    """Sanity check: an exact-match command is denied as expected."""
    rules = PermissionsConfig(deny={"RunCommand:rm -rf*"})
    tool = RunCommand(shell=mock_shell, permissions=PermissionEngine(rules))

    with pytest.raises(PermissionDeniedError):
        await tool.invoke(command="rm -rf /tmp/x")

    mock_shell.execute.assert_not_called()


async def test_whitespace_padded_command_still_denied(
    mock_shell: AsyncMock,
) -> None:
    """Leading/trailing whitespace cannot dodge RunCommand's DENY check.

    RunCommand.invoke() no longer builds its own key from the raw
    command; it delegates to PermissionEngine.enforce(), which strips
    via tool_call_key() before matching, so "  rm -rf /tmp/x" is keyed
    identically to "rm -rf /tmp/x".
    """
    rules = PermissionsConfig(deny={"RunCommand:rm -rf*"})
    tool = RunCommand(shell=mock_shell, permissions=PermissionEngine(rules))

    with pytest.raises(PermissionDeniedError):
        await tool.invoke(command="  rm -rf /tmp/x")

    mock_shell.execute.assert_not_called()


async def test_engine_evaluate_itself_is_whitespace_sensitive() -> None:
    """PermissionEngine.evaluate is a pure glob matcher with no
    normalization of its own -- by design, per its docstring, callers
    are responsible for canonicalizing the key first. This is why the
    fix lives in always routing key construction through the single
    tool_call_key() helper, not in evaluate() itself.
    """
    rules = PermissionsConfig(deny={"RunCommand:rm -rf*"})
    engine = PermissionEngine(rules)

    assert engine.evaluate("RunCommand:rm -rf /tmp/x") == PermissionLevel.DENY
    assert engine.evaluate("RunCommand:  rm -rf /tmp/x") != PermissionLevel.DENY
