"""Tests for the Fish shell plugin."""

import asyncio
import re
import shutil
import stat
import subprocess
import sys
import threading
from collections.abc import AsyncGenerator
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from agentsh.shell.plugin import fish as fish_module
from agentsh.shell.plugin.fish import (
    _SENTINEL,
    FishShell,
    _fish_quote,
    _parse_sentinel,
    _wrap_command,
)


class _FakeProcess:
    """Stand-in asyncio.subprocess.Process with preset stdout content."""

    def __init__(self, stdout_data: bytes, returncode: int | None = 0) -> None:
        self.stdout = asyncio.StreamReader()
        self.stdout.feed_data(stdout_data)
        self.stdout.feed_eof()
        self.returncode = returncode

    def kill(self) -> None:
        """Mark the process as terminated by kill."""
        self.returncode = -9

    async def wait(self) -> None:
        """No-op: nothing left to reap."""


def _make_shell(monkeypatch: pytest.MonkeyPatch) -> FishShell:
    """Build a FishShell with executable resolution stubbed out."""
    monkeypatch.setattr(
        fish_module.shutil, "which", lambda name: "/usr/bin/fish"
    )
    return FishShell()


def test_wrap_command_embeds_command_and_marker() -> None:
    """The wrapped script contains the raw command and a sentinel printf."""
    wrapped = _wrap_command("echo hello", "MARKER_1")
    assert "echo hello" in wrapped
    assert 'printf "%s:%d:%s\\n" "MARKER_1" "$__ec" "$(pwd)"' in wrapped
    assert "set __ec $status" in wrapped


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


def test_resolve_exe_raises_when_fish_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_resolve_exe raises RuntimeError when fish is not on PATH."""
    monkeypatch.setattr(fish_module.shutil, "which", lambda name: None)
    shell = FishShell()
    with pytest.raises(RuntimeError, match="fish"):
        shell._resolve_exe()


def test_resolve_exe_caches_result(monkeypatch: pytest.MonkeyPatch) -> None:
    """_resolve_exe only calls shutil.which once, caching the result."""
    which_mock = MagicMock(return_value="/usr/bin/fish")
    monkeypatch.setattr(fish_module.shutil, "which", which_mock)
    shell = FishShell()
    assert shell._resolve_exe() == "/usr/bin/fish"
    assert shell._resolve_exe() == "/usr/bin/fish"
    assert which_mock.call_count == 1


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


async def test_execute_full_round_trip_with_stderr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """execute wraps the command, parses the sentinel, and reads stderr."""
    shell = _make_shell(monkeypatch)

    def respond(script: str, marker: str) -> bytes:
        return f"hello\n{marker}:0:/home/x\n".encode()

    async def fake_spawn(exe: str, script: str, stderr_path: str) -> object:
        Path(stderr_path).write_text("oops\n")
        match = re.search(rf"{re.escape(_SENTINEL)}_[0-9a-f]+", script)
        assert match is not None
        return _FakeProcess(respond(script, match.group(0)))

    monkeypatch.setattr(shell, "_spawn", fake_spawn)
    result = await shell.execute("echo hello")
    assert result.stdout == "hello\n"
    assert result.stderr == "oops\n"
    assert result.exit_code == 0
    assert result.cwd == "/home/x"
    assert shell.cwd == "/home/x"


async def test_execute_nonzero_exit_code_round_trips(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A nonzero $status from the mocked subprocess is preserved."""
    shell = _make_shell(monkeypatch)

    async def fake_spawn(exe: str, script: str, stderr_path: str) -> object:
        match = re.search(rf"{re.escape(_SENTINEL)}_[0-9a-f]+", script)
        assert match is not None
        return _FakeProcess(f"{match.group(0)}:1:/home/x\n".encode())

    monkeypatch.setattr(shell, "_spawn", fake_spawn)
    result = await shell.execute("false")
    assert result.exit_code == 1


async def test_execute_embeds_command_in_script(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The command text is embedded verbatim in the spawned script."""
    shell = _make_shell(monkeypatch)
    seen_script = ""

    async def fake_spawn(exe: str, script: str, stderr_path: str) -> object:
        nonlocal seen_script
        seen_script = script
        match = re.search(rf"{re.escape(_SENTINEL)}_[0-9a-f]+", script)
        assert match is not None
        return _FakeProcess(f"hi\n{match.group(0)}:0:/home/x\n".encode())

    monkeypatch.setattr(shell, "_spawn", fake_spawn)
    result = await shell.execute("echo hi")
    assert "echo hi" in seen_script
    assert result.stdout == "hi\n"


async def test_execute_missing_sentinel_falls_back_to_returncode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the sentinel never appears, proc.returncode is used and cwd sticks.

    Simulates a command that calls `exit N` directly, bypassing the
    wrapper's sentinel printf.
    """
    shell = _make_shell(monkeypatch)
    original_cwd = shell.cwd

    async def fake_spawn(exe: str, script: str, stderr_path: str) -> object:
        return _FakeProcess(b"partial output\n", returncode=7)

    monkeypatch.setattr(shell, "_spawn", fake_spawn)
    result = await shell.execute("exit 7")
    assert result.exit_code == 7
    assert result.cwd == original_cwd
    assert result.stdout == "partial output\n"


async def test_execute_spawn_oserror_returns_error_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An OSError spawning the subprocess yields a graceful error result."""
    shell = _make_shell(monkeypatch)

    async def fake_spawn(exe: str, script: str, stderr_path: str) -> object:
        raise OSError("no such file")

    monkeypatch.setattr(shell, "_spawn", fake_spawn)
    result = await shell.execute("echo hi")
    assert result.exit_code == 1
    assert "no such file" in result.stderr


async def test_env_parses_name_value_pairs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """env() splits KEY=VALUE lines from the mocked `env` output."""
    shell = _make_shell(monkeypatch)

    async def fake_spawn(exe: str, script: str, stderr_path: str) -> object:
        match = re.search(rf"{re.escape(_SENTINEL)}_[0-9a-f]+", script)
        assert match is not None
        body = "PATH=/usr/bin\nHOME=/home/x\n"
        return _FakeProcess(f"{body}{match.group(0)}:0:/home/x\n".encode())

    monkeypatch.setattr(shell, "_spawn", fake_spawn)
    env = await shell.env()
    assert env == {"PATH": "/usr/bin", "HOME": "/home/x"}


async def test_complete_returns_completion_matches_without_descriptions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """complete() strips the tab-separated description from each match."""
    shell = _make_shell(monkeypatch)

    async def fake_spawn(exe: str, script: str, stderr_path: str) -> object:
        match = re.search(rf"{re.escape(_SENTINEL)}_[0-9a-f]+", script)
        assert match is not None
        body = "echo\tSend arguments to stdout\nenv\tRun a program\n"
        return _FakeProcess(f"{body}{match.group(0)}:0:/home/x\n".encode())

    monkeypatch.setattr(shell, "_spawn", fake_spawn)
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
    shell = _make_shell(monkeypatch)
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: _fake_completed_process(0)
    )
    assert await shell.can_parse("echo hi") is True


async def test_can_parse_false_when_parser_rejects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """can_parse is False when the mocked --no-execute check exits nonzero."""
    shell = _make_shell(monkeypatch)
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: _fake_completed_process(127)
    )
    assert await shell.can_parse(")(invalid((") is False


async def test_can_parse_false_on_subprocess_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """can_parse is False when the syntax-check subprocess times out."""
    shell = _make_shell(monkeypatch)

    def _raise(
        *args: object, **kwargs: object
    ) -> subprocess.CompletedProcess[bytes]:
        raise subprocess.TimeoutExpired(cmd="fish", timeout=1.0)

    monkeypatch.setattr(subprocess, "run", _raise)
    assert await shell.can_parse("echo hi") is False


async def test_can_parse_missing_fish_returns_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """can_parse returns False rather than raising when fish is missing."""
    monkeypatch.setattr(fish_module.shutil, "which", lambda name: None)
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


class _HangingFakeProcess:
    """Stand-in process whose stdout never delivers data or EOF.

    Unlike _FakeProcess (which always feeds EOF in __init__ and so
    can never actually hang), this stands in for a genuinely stuck
    command: read_until_sentinel blocks on it forever, exactly like a
    real hung fish process, until something cancels the awaiting task.
    """

    def __init__(self) -> None:
        self.stdout = asyncio.StreamReader()
        self.returncode: int | None = None

    def kill(self) -> None:
        """Mark the process as terminated by kill."""
        self.returncode = -9

    async def wait(self) -> None:
        """No-op: nothing left to reap."""


async def test_execute_cancellation_kills_in_flight_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cancelling execute() (e.g. via asyncio.wait_for) kills the subprocess."""
    shell = _make_shell(monkeypatch)
    proc = _HangingFakeProcess()

    async def fake_spawn(exe: str, script: str, stderr_path: str) -> object:
        return proc

    monkeypatch.setattr(shell, "_spawn", fake_spawn)
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(shell.execute("sleep 999"), timeout=0.05)
    assert proc.returncode == -9
    assert shell._current_proc is None


async def test_reset_kills_in_flight_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """reset() kills the current subprocess if one is in flight."""
    shell = _make_shell(monkeypatch)
    fake = _FakeProcess(b"", returncode=None)
    shell._current_proc = fake  # type: ignore[assignment]
    await shell.reset()
    assert fake.returncode == -9


async def test_reset_noop_without_in_flight_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """reset() is a no-op when no command is currently running."""
    shell = _make_shell(monkeypatch)
    await shell.reset()


async def test_close_delegates_to_reset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """close() kills an in-flight subprocess the same way reset() does."""
    shell = _make_shell(monkeypatch)
    fake = _FakeProcess(b"", returncode=None)
    shell._current_proc = fake  # type: ignore[assignment]
    await shell.close()
    assert fake.returncode == -9


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

    async def test_execute_does_not_hang_on_piped_stdin(
        self, shell: FishShell
    ) -> None:
        """A real fish command completes promptly rather than hanging.

        Regression test for #56: fish previously never ran anything fed
        over a piped stdin until EOF, so a persistent-process-backed
        execute() call hung forever. asyncio.wait_for with a short
        timeout turns that hang into a fast, clean test failure instead
        of a suite that never completes.
        """
        result = await asyncio.wait_for(shell.execute("echo hello"), timeout=5)
        assert result.stdout.strip() == "hello"

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
