"""Tests for permission key canonicalization in the agent loop."""

from pathlib import Path

import pytest

from agentsh.agent_loop import _tool_call_key
from agentsh.models import ToolCall


def test_write_file_key_resolves_relative_traversal() -> None:
    """Alternate spellings of the same file produce the same key."""
    tc = ToolCall(
        tool_name="WriteFile",
        arguments={"path": "./sub/../.env"},
        call_id="c1",
    )
    expected = (Path.cwd() / ".env").resolve().as_posix()
    assert _tool_call_key(tc) == f"WriteFile:{expected}"


def test_read_file_key_expands_home(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Tilde paths are keyed on the resolved absolute path."""
    monkeypatch.setenv("HOME", str(tmp_path))
    tc = ToolCall(
        tool_name="ReadFile",
        arguments={"path": "~/notes.txt"},
        call_id="c1",
    )
    expected = (tmp_path / "notes.txt").resolve().as_posix()
    assert _tool_call_key(tc) == f"ReadFile:{expected}"


def test_run_command_key_strips_whitespace() -> None:
    """Leading/trailing whitespace cannot dodge command glob patterns."""
    tc = ToolCall(
        tool_name="RunCommand",
        arguments={"command": "  rm -rf /tmp/x  "},
        call_id="c1",
    )
    assert _tool_call_key(tc) == "RunCommand:rm -rf /tmp/x"


def test_other_tool_key_is_bare_name() -> None:
    """Tools without path/command arguments key on the tool name."""
    tc = ToolCall(tool_name="WebFetch", arguments={}, call_id="c1")
    assert _tool_call_key(tc) == "WebFetch"
