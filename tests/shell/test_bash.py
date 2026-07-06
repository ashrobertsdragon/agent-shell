"""Integration tests for BashShell against a real bash subprocess."""

import os
from collections.abc import AsyncGenerator
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agentsh.shell.plugin import bash as bash_module
from agentsh.shell.plugin.bash import BashShell


class _FakeClock:
    """Stand-in for the time module yielding preset monotonic ticks."""

    def __init__(self, *ticks: float) -> None:
        self._ticks = iter(ticks)

    def monotonic(self) -> float:
        """Return the next preset tick."""
        return next(self._ticks)


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
    cwd = shell.cwd
    assert cwd == "/tmp"


async def test_can_parse_valid(shell: BashShell) -> None:
    """can_parse returns True for valid shell syntax."""
    assert await shell.can_parse("ls -la") is True


async def test_can_parse_invalid(shell: BashShell) -> None:
    """can_parse returns False for invalid shell syntax."""
    assert await shell.can_parse(")(invalid((") is False


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


async def test_execute_duration_unit_consistent_on_child_process_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The ChildProcessError branch reports duration in milliseconds."""
    s = BashShell()
    s._process = SimpleNamespace(stdin=None, returncode=None)  # type: ignore[assignment]
    monkeypatch.setattr(bash_module, "time", _FakeClock(0.0, 0.5))
    result = await s.execute("true")
    assert result.duration_ms == 500.0
    assert result.exit_code == 1
