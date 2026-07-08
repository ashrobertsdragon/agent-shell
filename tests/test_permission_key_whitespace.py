"""Regression tests for whitespace handling in permission-key construction.

There are two independent places that build a ``"RunCommand:{command}"``
permission key:

- ``agent_loop._tool_call_key`` strips the command before keying it (see
  ``tests/test_permission_keys.py::test_run_command_key_strips_whitespace``),
  so the agent loop's own DENY/CONFIRM gate is immune to whitespace padding.
- ``RunCommand._check_key`` (src/agentsh/tools/run_command.py) builds the
  same style of key directly from the raw ``command`` argument, with no
  stripping.

Because ``fnmatch`` patterns like ``"RunCommand:rm -rf*"`` do not match a
key with leading whitespace (``"RunCommand:  rm -rf ..."``), a command
padded with leading/trailing whitespace can defeat RunCommand's own
internal DENY check even though the equivalent, correctly-stripped key
would have matched. In normal interactive use this is masked because
both ``repl.py`` (strips before building the key) and ``run_agent_loop``
(strips via ``_tool_call_key``) gate the call *before* invoke() ever
runs — but ``RunCommand.invoke`` is a public method or a defense-in-depth
check, and calling it directly with a padded command bypasses those
outer gates.

These tests pin the *current* behavior rather than "fixing" it, since
tools/run_command.py's enforcement logic is out of scope here (owned by
a parallel in-flight change). See the task report for the recommended
fix: normalize the command the same way ``_tool_call_key`` does before
building the key in ``RunCommand._check_key``.
"""

from unittest.mock import AsyncMock

import pytest

from agentsh.config import PermissionRulesConfig
from agentsh.permissions import PermissionEngine, PermissionLevel
from agentsh.tools.run_command import PermissionDeniedError, RunCommand


@pytest.fixture
def mock_shell() -> AsyncMock:
    """Shell mock whose execute() we can assert was (not) reached."""
    return AsyncMock()


async def test_stripped_command_is_denied(mock_shell: AsyncMock) -> None:
    """Sanity check: an exact-match command is denied as expected."""
    rules = PermissionRulesConfig(deny={"RunCommand:rm -rf*"})
    tool = RunCommand(shell=mock_shell, permissions=PermissionEngine(rules))

    with pytest.raises(PermissionDeniedError):
        await tool.invoke(command="rm -rf /tmp/x")

    mock_shell.execute.assert_not_called()


async def test_whitespace_padded_command_bypasses_deny_rule(
    mock_shell: AsyncMock,
) -> None:
    """KNOWN DEFECT: leading whitespace defeats RunCommand's DENY check.

    ``RunCommand._check_key`` keys on the raw, unstripped command, so
    ``"  rm -rf /tmp/x"`` produces the key
    ``"RunCommand:  rm -rf /tmp/x"``, which the deny glob
    ``"RunCommand:rm -rf*"`` does not match. The command is executed
    instead of being blocked, even though the semantically identical
    stripped command is correctly denied (see the test above).
    """
    rules = PermissionRulesConfig(deny={"RunCommand:rm -rf*"})
    tool = RunCommand(shell=mock_shell, permissions=PermissionEngine(rules))

    await tool.invoke(command="  rm -rf /tmp/x")

    mock_shell.execute.assert_called_once_with("  rm -rf /tmp/x")


async def test_engine_evaluate_itself_is_whitespace_sensitive() -> None:
    """The mismatch is a key-construction bug, not a PermissionEngine bug.

    PermissionEngine.evaluate is a pure glob matcher with no normalization
    of its own — by design, per its docstring, callers are responsible for
    canonicalizing the key. Both call sites agree that the *canonical* key
    strips the command; only RunCommand._check_key fails to do so.
    """
    rules = PermissionRulesConfig(deny={"RunCommand:rm -rf*"})
    engine = PermissionEngine(rules)

    assert engine.evaluate("RunCommand:rm -rf /tmp/x") == PermissionLevel.DENY
    assert engine.evaluate("RunCommand:  rm -rf /tmp/x") != PermissionLevel.DENY
