"""Tests for the Fish shell plugin."""

import asyncio
import re
import shutil
import stat
import subprocess
import sys
import threading
from collections.abc import AsyncGenerator, Callable
from pathlib import Path
from types import SimpleNamespace

import pytest

from agentsh.shell.plugin import fish as fish_module
from agentsh.shell.plugin.fish import (
    _SENTINEL,
    FishShell,
    _fish_quote,
    _parse_sentinel,
)


class _FakeClock:
    """Stand-in for the time module yielding preset monotonic ticks."""

    def __init__(self, *ticks: float) -> None:
        self._ticks = iter(ticks)

    def monotonic(self) -> float:
        """Return the next preset tick."""
        return next(self._ticks)


class _FakeStdin:
    """Synthesizes a fish-shaped stdout reply for each write.

    execute() writes the begin/end-wrapped command plus the
    sentinel-print statement in a single call, then reads stdout until
    the sentinel line appears. This fake extracts the per-call marker
    from that write and feeds a canned reply straight into the paired
    StreamReader, so execute()'s wrapping and sentinel-parsing logic
    can be exercised without a real fish subprocess.
    """

    def __init__(
        self,
        stdout: asyncio.StreamReader,
        respond: Callable[[str, str], str],
    ) -> None:
        self._stdout = stdout
        self._respond = respond

    def write(self, data: bytes) -> None:
        """Extract the marker from the write and feed the canned reply."""
        text = data.decode()
        match = re.search(rf"{re.escape(_SENTINEL)}_[0-9a-f]+", text)
        assert match is not None, "wrapped command must embed a marker"
        self._stdout.feed_data(self._respond(text, match.group(0)).encode())

    async def drain(self) -> None:
        """No-op: write() already delivered the reply synchronously."""


class _FakeProcess:
    """Stand-in asyncio.subprocess.Process driven by a canned responder."""

    def __init__(self, respond: Callable[[str, str], str]) -> None:
        self.stdout = asyncio.StreamReader()
        self.stdin = _FakeStdin(self.stdout, respond)
        self.returncode: int | None = None

    def kill(self) -> None:
        """Mark the process as terminated by kill."""
        self.returncode = -9

    def terminate(self) -> None:
        """Mark the process as terminated, mirroring Process.terminate()."""
        self.returncode = -15

    async def wait(self) -> None:
        """No-op: nothing left to reap."""


def test_parse_sentinel_matches_exact_marker() -> None:
    """A well-formed line for the given marker parses to (code, cwd)."""
    marker = f"{_SENTINEL}_nonce"
    assert _parse_sentinel(f"{marker}:0:/tmp\n", marker) == (0, "/tmp")


def test_parse_sentinel_rejects_different_nonce() -> None:
    """A line carrying a different call's nonce is not a match."""
    marker = f"{_SENTINEL}_nonce-a"
    other = f"{_SENTINEL}_nonce-b"
    assert _parse_sentinel(f"{other}:0:/tmp\n", marker) is None


def test_parse_sentinel_rejects_prefix_only_match() -> None:
    """A line that merely starts with the marker text is not a match."""
    marker = f"{_SENTINEL}_nonce"
    assert _parse_sentinel(f"{marker}extra:0:/tmp\n", marker) is None


def test_parse_sentinel_rejects_malformed_line() -> None:
    """A line missing the code/cwd fields returns None instead of raising."""
    marker = f"{_SENTINEL}_nonce"
    assert _parse_sentinel(f"{marker}:not-an-int:/tmp\n", marker) is None
    assert _parse_sentinel("unrelated output\n", marker) is None


def test_parse_sentinel_strips_carriage_return() -> None:
    """A trailing \\r\\n does not break the cwd field."""
    marker = f"{_SENTINEL}_nonce"
    assert _parse_sentinel(f"{marker}:0:/tmp\r\n", marker) == (0, "/tmp")


def test_fish_quote_escapes_single_quotes_and_backslashes() -> None:
    """Embedded single quotes and backslashes are escaped, not the rest."""
    assert _fish_quote("it's") == r"'it\'s'"
    assert _fish_quote("a\\b") == r"'a\\b'"
    assert _fish_quote("plain") == "'plain'"


def _patch_default_history_path(
    monkeypatch: pytest.MonkeyPatch, path: Path
) -> None:
    """Point agentsh's own fish history file at path."""
    monkeypatch.setattr(fish_module, "_default_history_path", lambda: path)


async def test_history_round_trip(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """append_history then history round-trips without a subprocess."""
    _patch_default_history_path(monkeypatch, tmp_path / "sub" / "hist.txt")
    shell = FishShell()
    await shell.append_history("echo one")
    await shell.append_history("echo two")
    assert await shell.history() == ["echo one", "echo two"]
    assert await shell.history(limit=1) == ["echo two"]


async def test_history_missing_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """history returns an empty list when the file does not exist."""
    _patch_default_history_path(monkeypatch, tmp_path / "missing.txt")
    shell = FishShell()
    assert await shell.history() == []


async def test_append_history_writes_own_secure_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """append_history writes agentsh's own file at mode 0o600."""
    own_file = tmp_path / "fish_history"
    _patch_default_history_path(monkeypatch, own_file)
    shell = FishShell()
    await shell.append_history("echo hi")
    assert own_file.read_text() == "echo hi\n"
    if sys.platform != "win32":
        assert stat.S_IMODE(own_file.stat().st_mode) == 0o600


async def test_execute_full_round_trip_with_stderr() -> None:
    """execute wraps the command, parses the sentinel, and reads stderr.

    Drives the real execute() implementation against a mocked fish
    subprocess so the begin/end wrapping, $status capture, and
    sentinel/stderr handling is validated without a real fish binary.
    """

    def respond(wrapped: str, marker: str) -> str:
        stderr_match = re.search(r"end 2>(\S+)", wrapped)
        assert stderr_match is not None
        Path(stderr_match.group(1)).write_text("oops\n")
        assert "$status" in wrapped
        return f"hello\n{marker}:0:/home/x\n"

    shell = FishShell()
    shell._process = _FakeProcess(respond)  # type: ignore[assignment]
    result = await shell.execute("echo hello")
    assert result.stdout == "hello\n"
    assert result.stderr == "oops\n"
    assert result.exit_code == 0
    assert result.cwd == "/home/x"
    assert shell.cwd == "/home/x"


async def test_execute_nonzero_exit_code_round_trips() -> None:
    """A nonzero $status from the mocked subprocess is preserved."""

    def respond(wrapped: str, marker: str) -> str:
        return f"{marker}:1:/home/x\n"

    shell = FishShell()
    shell._process = _FakeProcess(respond)  # type: ignore[assignment]
    result = await shell.execute("false")
    assert result.exit_code == 1


async def test_execute_wraps_in_begin_end_block() -> None:
    """The command body is wrapped in a begin/end block, not braces."""

    def respond(wrapped: str, marker: str) -> str:
        assert wrapped.startswith("begin\n")
        assert "echo hi" in wrapped
        return f"hi\n{marker}:0:/home/x\n"

    shell = FishShell()
    shell._process = _FakeProcess(respond)  # type: ignore[assignment]
    result = await shell.execute("echo hi")
    assert result.stdout == "hi\n"


async def test_execute_duration_unit_consistent_on_child_process_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The ChildProcessError branch reports duration in milliseconds."""
    shell = FishShell()
    shell._process = SimpleNamespace(  # type: ignore[assignment]
        stdin=None, returncode=None
    )
    monkeypatch.setattr(fish_module, "time", _FakeClock(0.0, 0.5))
    result = await shell.execute("true")
    assert result.duration_ms == 500.0
    assert result.exit_code == 1


async def test_env_parses_name_value_pairs() -> None:
    """env() splits KEY=VALUE lines from the mocked `env` output."""

    def respond(wrapped: str, marker: str) -> str:
        body = "PATH=/usr/bin\nHOME=/home/x\n"
        return f"{body}{marker}:0:/home/x\n"

    shell = FishShell()
    shell._process = _FakeProcess(respond)  # type: ignore[assignment]
    env = await shell.env()
    assert env == {"PATH": "/usr/bin", "HOME": "/home/x"}


async def test_complete_returns_completion_matches_without_descriptions() -> (
    None
):
    """complete() strips the tab-separated description from each match."""

    def respond(wrapped: str, marker: str) -> str:
        body = "echo\tSend arguments to stdout\nenv\tRun a program\n"
        return f"{body}{marker}:0:/home/x\n"

    shell = FishShell()
    shell._process = _FakeProcess(respond)  # type: ignore[assignment]
    matches = await shell.complete("e")
    assert matches == ["echo", "env"]


def _fake_completed_process(
    returncode: int,
) -> subprocess.CompletedProcess[bytes]:
    """Build a minimal CompletedProcess carrying only a returncode."""
    return subprocess.CompletedProcess(args=[], returncode=returncode)


async def test_can_parse_true_when_parser_accepts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """can_parse is True when the mocked --no-execute check exits zero."""
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: _fake_completed_process(0)
    )
    shell = FishShell()
    assert await shell.can_parse("echo hi") is True


async def test_can_parse_false_when_parser_rejects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """can_parse is False when the mocked --no-execute check exits nonzero."""
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: _fake_completed_process(127)
    )
    shell = FishShell()
    assert await shell.can_parse(")(invalid((") is False


async def test_can_parse_false_on_subprocess_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """can_parse is False when the syntax-check subprocess times out."""

    def _raise(
        *args: object, **kwargs: object
    ) -> subprocess.CompletedProcess[bytes]:
        raise subprocess.TimeoutExpired(cmd="fish", timeout=1.0)

    monkeypatch.setattr(subprocess, "run", _raise)
    shell = FishShell()
    assert await shell.can_parse("echo hi") is False


async def test_history_read_runs_off_the_event_loop(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """history() reads the own history file via asyncio.to_thread, not inline."""
    _patch_default_history_path(monkeypatch, tmp_path / "hist.txt")
    shell = FishShell()
    await shell.append_history("echo one")

    main_thread = threading.current_thread()
    read_thread: threading.Thread | None = None
    original_read_last_lines = fish_module.read_last_lines

    def _spy(path: Path, limit: int) -> list[str]:
        nonlocal read_thread
        read_thread = threading.current_thread()
        return original_read_last_lines(path, limit)

    monkeypatch.setattr(fish_module, "read_last_lines", _spy)
    await shell.history()

    assert read_thread is not None
    assert read_thread is not main_thread


async def test_append_history_write_runs_off_the_event_loop(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """append_history's own-file write runs via asyncio.to_thread, not inline."""
    _patch_default_history_path(monkeypatch, tmp_path / "hist.txt")
    shell = FishShell()
    main_thread = threading.current_thread()
    write_thread: threading.Thread | None = None
    original = fish_module.append_secure_line

    def _spy(path: Path, line: str) -> None:
        nonlocal write_thread
        write_thread = threading.current_thread()
        original(path, line)

    monkeypatch.setattr(fish_module, "append_secure_line", _spy)
    await shell.append_history("echo tracked")

    assert write_thread is not None
    assert write_thread is not main_thread


requires_fish = pytest.mark.skipif(
    shutil.which("fish") is None, reason="fish not installed"
)


@requires_fish
class TestFishIntegration:
    """Integration tests against a real fish subprocess."""

    @pytest.fixture
    async def shell(self) -> AsyncGenerator[FishShell, None]:
        """Yield a FishShell and ensure it's closed after each test."""
        s = FishShell()
        yield s
        await s.close()

    async def test_execute_echo(self, shell: FishShell) -> None:
        """Execute captures stdout."""
        result = await shell.execute("echo hello")
        assert result.stdout.strip() == "hello"
        assert result.exit_code == 0

    async def test_execute_captures_stderr(self, shell: FishShell) -> None:
        """Execute captures stderr separately."""
        result = await shell.execute("echo err >&2")
        assert "err" in result.stderr
        assert result.exit_code == 0

    async def test_execute_tracks_exit_code(self, shell: FishShell) -> None:
        """Execute returns the last exit code."""
        result = await shell.execute("false")
        assert result.exit_code == 1

    async def test_execute_tracks_cwd(self, shell: FishShell) -> None:
        """cwd reflects directory changes made by cd."""
        await shell.execute("cd /tmp")
        result = await shell.execute("pwd")
        assert result.stdout.strip() == "/tmp"
        assert shell.cwd == "/tmp"

    async def test_execute_multiline_block(self, shell: FishShell) -> None:
        """A multi-line fish construct executes as a single unit."""
        result = await shell.execute("if true\n    echo block\nend")
        assert result.stdout.strip() == "block"
        assert result.exit_code == 0

    async def test_can_parse_valid(self, shell: FishShell) -> None:
        """can_parse returns True for valid fish syntax."""
        assert await shell.can_parse("echo hi") is True

    async def test_can_parse_invalid(self, shell: FishShell) -> None:
        """can_parse returns False for invalid fish syntax."""
        assert await shell.can_parse(")(invalid((") is False

    async def test_env_contains_path(self, shell: FishShell) -> None:
        """env returns the subprocess environment."""
        env = await shell.env()
        assert "PATH" in env

    async def test_complete_returns_matches(self, shell: FishShell) -> None:
        """complete returns candidates for a command prefix."""
        matches = await shell.complete("ech")
        assert any("echo" in m for m in matches)

    async def test_render_prompt_returns_nonempty(
        self, shell: FishShell
    ) -> None:
        """render_prompt returns a non-empty string."""
        prompt = await shell.render_prompt()
        assert isinstance(prompt, str)
        assert len(prompt) > 0
