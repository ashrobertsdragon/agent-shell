"""Tests for the shared tool_call_key permission-key builder."""

from pathlib import Path

import pytest

from agentsh.permissions import tool_call_key


def test_write_file_key_resolves_relative_traversal() -> None:
    """Alternate spellings of the same file produce the same key."""
    key = tool_call_key("WriteFile", {"path": "./sub/../.env"})
    expected = (Path.cwd() / ".env").resolve().as_posix()
    assert key == f"WriteFile:{expected}"


def test_read_file_key_expands_home(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Tilde paths are keyed on the resolved absolute path."""
    monkeypatch.setenv("HOME", str(tmp_path))
    key = tool_call_key("ReadFile", {"path": "~/notes.txt"})
    expected = (tmp_path / "notes.txt").resolve().as_posix()
    assert key == f"ReadFile:{expected}"


def test_run_command_key_strips_whitespace() -> None:
    """Leading/trailing whitespace cannot dodge command glob patterns."""
    key = tool_call_key("RunCommand", {"command": "  rm -rf /tmp/x  "})
    assert key == "RunCommand:rm -rf /tmp/x"


def test_other_tool_key_is_bare_name() -> None:
    """Tools without path/command arguments key on the tool name."""
    assert tool_call_key("WebFetch", {}) == "WebFetch"
