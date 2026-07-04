"""Integration tests for BashShell against a real bash subprocess."""

from __future__ import annotations

import os
from collections.abc import AsyncGenerator
from pathlib import Path
from unittest.mock import patch

import pytest

from agentsh.shell.bash import BashShell


@pytest.fixture
async def shell() -> AsyncGenerator[BashShell, None]:
    """Yield a BashShell and ensure it's closed after each test."""
    s = BashShell()
    yield s
    await s.close()


async def test_execute_echo(shell: BashShell) -> None:
    """Execute captures stdout."""
    result = await shell.execute("echo hello")
    assert result.stdout.strip() == "hello"
    assert result.exit_code == 0


async def test_execute_captures_stderr(shell: BashShell) -> None:
    """Execute captures stderr separately."""
    result = await shell.execute("echo err >&2")
    assert "err" in result.stderr
    assert result.exit_code == 0


async def test_execute_tracks_exit_code(shell: BashShell) -> None:
    """Execute returns the last exit code."""
    result = await shell.execute("false")
    assert result.exit_code == 1


async def test_execute_tracks_cwd(shell: BashShell) -> None:
    """cwd() reflects directory changes made by cd."""
    await shell.execute("cd /tmp")
    result = await shell.execute("pwd")
    assert result.stdout.strip() == "/tmp"
    cwd = await shell.cwd()
    assert cwd == "/tmp"


async def test_can_parse_valid(shell: BashShell) -> None:
    """can_parse returns True for valid shell syntax."""
    assert shell.can_parse("ls -la") is True


async def test_can_parse_invalid(shell: BashShell) -> None:
    """can_parse returns False for invalid shell syntax."""
    assert shell.can_parse(")(invalid((") is False


async def test_render_prompt_returns_nonempty(shell: BashShell) -> None:
    """render_prompt returns a non-empty string."""
    prompt = await shell.render_prompt()
    assert isinstance(prompt, str)
    assert len(prompt) > 0


async def test_append_history_writes_to_histfile(
    shell: BashShell, tmp_path: Path
) -> None:
    """append_history writes to the file pointed to by $HISTFILE."""
    histfile = str(tmp_path / ".bash_history")
    with patch.dict(os.environ, {"HISTFILE": histfile}):
        await shell.append_history("ls -la")
    assert "ls -la" in (tmp_path / ".bash_history").read_text()
