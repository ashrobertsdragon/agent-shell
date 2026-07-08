"""Integration tests for BashShell against a real bash subprocess."""

import os
import shlex
from collections.abc import AsyncGenerator
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agentsh.shell.plugin import bash as bash_module
from agentsh.shell.plugin._base import new_marker
from agentsh.shell.plugin.bash import BashShell, _parse_sentinel


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


def test_parse_sentinel_matches_exact_marker() -> None:
    """A well-formed line for the given marker parses to (code, cwd)."""
    marker = new_marker(bash_module._SENTINEL)
    assert _parse_sentinel(f"{marker}:0:/tmp\n", marker) == (0, "/tmp")


def test_parse_sentinel_rejects_different_nonce() -> None:
    """A line carrying a different call's nonce is not a match.

    This is the regression case for sentinel spoofing: two calls to
    `new_marker` never collide, so output from one command cannot be
    mistaken for another's completion line.
    """
    marker = new_marker(bash_module._SENTINEL)
    other = new_marker(bash_module._SENTINEL)
    assert marker != other
    assert _parse_sentinel(f"{other}:0:/tmp\n", marker) is None


def test_parse_sentinel_rejects_prefix_only_match() -> None:
    """A line that merely starts with the marker text is not a match."""
    marker = new_marker(bash_module._SENTINEL)
    assert _parse_sentinel(f"{marker}extra:0:/tmp\n", marker) is None


def test_parse_sentinel_rejects_malformed_line() -> None:
    """A line missing the code/cwd fields returns None instead of raising."""
    marker = new_marker(bash_module._SENTINEL)
    assert _parse_sentinel(f"{marker}:not-an-int:/tmp\n", marker) is None
    assert _parse_sentinel("unrelated output\n", marker) is None


def test_parse_sentinel_strips_carriage_return() -> None:
    """A trailing \\r\\n (as from a PTY) does not break the cwd field."""
    marker = new_marker(bash_module._SENTINEL)
    assert _parse_sentinel(f"{marker}:0:/tmp\r\n", marker) == (0, "/tmp")


async def test_execute_survives_sentinel_lookalike_output(
    shell: BashShell,
) -> None:
    """Command output containing a sentinel-lookalike line does not desync.

    This is the end-to-end regression test for issue #10's sentinel
    spoofing bug: a command that prints text shaped exactly like the
    completion sentinel (but with a forged nonce) must not be mistaken
    for the real one, and the next command must still execute cleanly.
    """
    forged = f"{bash_module._SENTINEL}_forged-nonce:0:/forged/path"
    result = await shell.execute(f"echo {shlex.quote(forged)}")
    assert result.stdout.strip() == forged
    assert result.exit_code == 0

    follow_up = await shell.execute("echo still-in-sync")
    assert follow_up.stdout.strip() == "still-in-sync"
    assert follow_up.exit_code == 0


async def test_process_restarts_when_desynced_even_if_alive(
    shell: BashShell,
) -> None:
    """process restarts on the desynced flag, not just on process death."""
    first = await shell.process
    assert first.returncode is None
    shell._desynced = True
    second = await shell.process
    assert second is not first
    assert shell._desynced is False


async def test_reset_kills_process_and_forces_restart(
    shell: BashShell,
) -> None:
    """reset kills the live subprocess and the next `process` access restarts it."""
    first = await shell.process
    assert first.returncode is None
    await shell.reset()
    assert first.returncode is not None
    second = await shell.process
    assert second is not first
    assert second.returncode is None
