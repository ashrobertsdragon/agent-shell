"""Tests for the Nushell shell plugin."""

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

from agentsh.shell.plugin import nushell as nushell_module
from agentsh.shell.plugin.nushell import (
    _SENTINEL,
    NuShellShell,
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


def _make_shell(monkeypatch: pytest.MonkeyPatch) -> NuShellShell:
    """Build a NuShellShell with executable resolution stubbed out."""
    monkeypatch.setattr(
        nushell_module.shutil, "which", lambda name: "/usr/bin/nu"
    )
    return NuShellShell()


def test_wrap_command_embeds_command_and_marker() -> None:
    """The wrapped script contains the raw command and a sentinel print."""
    wrapped = _wrap_command("print hello", "MARKER_1")
    assert "print hello" in wrapped
    assert 'print $"MARKER_1:($env.LAST_EXIT_CODE):($env.PWD)"' in wrapped
    assert "$env.LAST_EXIT_CODE = 0" in wrapped
    assert "catch {|err|" in wrapped


def test_parse_sentinel_basic() -> None:
    """A well-formed sentinel line yields (exit_code, cwd)."""
    marker = f"{_SENTINEL}_nonce"
    assert _parse_sentinel(f"{marker}:0:/home/x\n", marker) == (
        0,
        "/home/x",
    )


def test_parse_sentinel_cwd_with_colon() -> None:
    """maxsplit=2 keeps a colon-containing cwd intact as the last field."""
    marker = f"{_SENTINEL}_nonce"
    parsed = _parse_sentinel(f"{marker}:1:C:\\Users\\x\n", marker)
    assert parsed == (1, "C:\\Users\\x")


def test_parse_sentinel_rejects_lookalike_without_matching_marker() -> None:
    """A line that merely starts with the sentinel text is not a match."""
    marker = f"{_SENTINEL}_nonce-a"
    other_marker = f"{_SENTINEL}_nonce-b"
    assert _parse_sentinel(f"{other_marker}:0:/tmp\n", marker) is None


def test_parse_sentinel_rejects_malformed_line() -> None:
    """A line missing the code/cwd fields returns None instead of raising."""
    marker = f"{_SENTINEL}_nonce"
    assert _parse_sentinel(f"{marker}:not-an-int:/tmp\n", marker) is None
    assert _parse_sentinel("unrelated output\n", marker) is None


def test_resolve_exe_raises_when_nu_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_resolve_exe raises RuntimeError when nu is not on PATH."""
    monkeypatch.setattr(nushell_module.shutil, "which", lambda name: None)
    shell = NuShellShell()
    with pytest.raises(RuntimeError, match="nu"):
        shell._resolve_exe()


def test_resolve_exe_caches_result(monkeypatch: pytest.MonkeyPatch) -> None:
    """_resolve_exe only calls shutil.which once, caching the result."""
    which_mock = MagicMock(return_value="/usr/bin/nu")
    monkeypatch.setattr(nushell_module.shutil, "which", which_mock)
    shell = NuShellShell()
    assert shell._resolve_exe() == "/usr/bin/nu"
    assert shell._resolve_exe() == "/usr/bin/nu"
    assert which_mock.call_count == 1


async def test_execute_full_round_trip_with_stderr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """execute wraps the command, parses the sentinel, and reads stderr."""
    shell = _make_shell(monkeypatch)

    def respond(script: str, marker: str) -> bytes:
        return f"hello\n{marker}:0:/tmp/nu-cwd\n".encode()

    async def fake_spawn(exe: str, script: str, stderr_path: str) -> object:
        Path(stderr_path).write_text("oops\n")
        match = re.search(rf"{re.escape(_SENTINEL)}_[0-9a-f]+", script)
        assert match is not None
        return _FakeProcess(respond(script, match.group(0)))

    monkeypatch.setattr(shell, "_spawn", fake_spawn)
    result = await shell.execute("print hello")
    assert result.stdout == "hello\n"
    assert result.stderr == "oops\n"
    assert result.exit_code == 0
    assert result.cwd == "/tmp/nu-cwd"
    assert shell.cwd == "/tmp/nu-cwd"


async def test_execute_nonzero_exit_code_round_trips(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A nonzero $env.LAST_EXIT_CODE from the mocked subprocess is preserved."""
    shell = _make_shell(monkeypatch)

    async def fake_spawn(exe: str, script: str, stderr_path: str) -> object:
        match = re.search(rf"{re.escape(_SENTINEL)}_[0-9a-f]+", script)
        assert match is not None
        return _FakeProcess(f"{match.group(0)}:1:/tmp\n".encode())

    monkeypatch.setattr(shell, "_spawn", fake_spawn)
    result = await shell.execute("exit 1")
    assert result.exit_code == 1
    assert result.cwd == "/tmp"


async def test_execute_missing_sentinel_falls_back_to_returncode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the sentinel never appears, proc.returncode is used and cwd sticks.

    Simulates a command that calls `exit N` directly, bypassing the
    wrapper's try/catch and its sentinel print.
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
    result = await shell.execute("print hi")
    assert result.exit_code == 1
    assert "no such file" in result.stderr


async def test_execute_caught_error_reports_exit_code_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The catch branch's exit_code=1 convention round-trips through execute."""
    shell = _make_shell(monkeypatch)

    async def fake_spawn(exe: str, script: str, stderr_path: str) -> object:
        match = re.search(rf"{re.escape(_SENTINEL)}_[0-9a-f]+", script)
        assert match is not None
        Path(stderr_path).write_text("some nushell error\n")
        return _FakeProcess(f"{match.group(0)}:1:/tmp\n".encode())

    monkeypatch.setattr(shell, "_spawn", fake_spawn)
    result = await shell.execute("1 / 0")
    assert result.exit_code == 1
    assert result.stderr == "some nushell error\n"


async def test_env_parses_json_and_keeps_only_strings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """env() parses `$env | to json -r` output, dropping non-string values."""
    from agentsh.models import CommandResult

    shell = _make_shell(monkeypatch)

    async def fake_execute(command: str) -> CommandResult:
        assert command == "$env | to json -r"
        return CommandResult(
            stdout='{"FOO":"bar","COUNT":1,"OK":true}',
            stderr="",
            exit_code=0,
            duration_ms=1.0,
            cwd=shell.cwd,
        )

    monkeypatch.setattr(shell, "execute", fake_execute)
    env = await shell.env()
    assert env == {"FOO": "bar"}


async def test_env_handles_invalid_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """env() returns an empty dict rather than raising on bad JSON."""
    from agentsh.models import CommandResult

    shell = _make_shell(monkeypatch)

    async def fake_execute(command: str) -> CommandResult:
        return CommandResult(
            stdout="not json",
            stderr="",
            exit_code=1,
            duration_ms=1.0,
            cwd=shell.cwd,
        )

    monkeypatch.setattr(shell, "execute", fake_execute)
    assert await shell.env() == {}


async def test_complete_finds_path_executables(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """complete() matches executables on PATH by prefix without spawning nu."""
    (tmp_path / "gitthing").touch(mode=0o755)
    (tmp_path / "notes.txt").touch(mode=0o644)
    monkeypatch.setenv("PATH", str(tmp_path))
    shell = _make_shell(monkeypatch)
    matches = await shell.complete("git")
    assert "gitthing" in matches
    assert "notes.txt" not in matches


async def test_complete_skips_missing_path_dirs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Nonexistent PATH entries are skipped without error."""
    monkeypatch.setenv("PATH", "/nonexistent-agentsh-dir")
    shell = _make_shell(monkeypatch)
    assert await shell.complete("x") == []


async def test_can_parse_valid_script(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """can_parse returns True when the mocked nu-check run exits 0."""
    shell = _make_shell(monkeypatch)
    monkeypatch.setattr(
        nushell_module.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(a, 0),
    )
    assert await shell.can_parse("print hello") is True


async def test_can_parse_invalid_script(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """can_parse returns False when the mocked nu-check run exits nonzero."""
    shell = _make_shell(monkeypatch)
    monkeypatch.setattr(
        nushell_module.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(a, 1),
    )
    assert await shell.can_parse(")((nonsense") is False


async def test_can_parse_missing_nu_returns_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """can_parse returns False rather than raising when nu is missing."""
    monkeypatch.setattr(nushell_module.shutil, "which", lambda name: None)
    shell = NuShellShell()
    assert await shell.can_parse("print hi") is False


async def test_can_parse_timeout_returns_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A timed-out nu-check invocation is treated as unparsable."""
    shell = _make_shell(monkeypatch)

    def _raise(
        *args: object, **kwargs: object
    ) -> subprocess.CompletedProcess[bytes]:
        raise subprocess.TimeoutExpired(cmd="nu", timeout=2.0)

    monkeypatch.setattr(nushell_module.subprocess, "run", _raise)
    assert await shell.can_parse("print hi") is False


async def test_render_prompt_is_synthesized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """render_prompt returns a synthesized `cwd> ` string."""
    shell = _make_shell(monkeypatch)
    assert await shell.render_prompt() == f"{shell.cwd}> "


async def test_history_round_trip(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """append_history then history round-trips via the own file."""
    shell = _make_shell(monkeypatch)
    shell._history_path = tmp_path / "agentsh" / "nushell_history"
    await shell.append_history("ls")
    await shell.append_history("cd ..")
    assert await shell.history() == ["ls", "cd .."]
    assert await shell.history(limit=1) == ["cd .."]


async def test_history_missing_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """history returns an empty list when the file does not exist."""
    shell = _make_shell(monkeypatch)
    shell._history_path = tmp_path / "agentsh" / "nushell_history"
    assert await shell.history() == []


async def test_append_history_writes_own_secure_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """append_history writes the own history file at mode 0o600."""
    shell = _make_shell(monkeypatch)
    shell._history_path = tmp_path / "agentsh" / "nushell_history"
    await shell.append_history("ls")
    assert shell._history_path.read_text() == "ls\n"
    if sys.platform != "win32":
        assert stat.S_IMODE(shell._history_path.stat().st_mode) == 0o600


async def test_history_read_runs_off_the_event_loop(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """history() reads the own history file via asyncio.to_thread, not inline."""
    shell = _make_shell(monkeypatch)
    shell._history_path = tmp_path / "agentsh" / "nushell_history"
    await shell.append_history("ls")

    main_thread = threading.current_thread()
    read_thread: threading.Thread | None = None
    original_read_last_lines = nushell_module.read_last_lines

    def _spy(path: Path, limit: int) -> list[str]:
        nonlocal read_thread
        read_thread = threading.current_thread()
        return original_read_last_lines(path, limit)

    monkeypatch.setattr(nushell_module, "read_last_lines", _spy)
    await shell.history()

    assert read_thread is not None
    assert read_thread is not main_thread


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


requires_nu = pytest.mark.skipif(
    shutil.which("nu") is None, reason="nu is not installed"
)


@requires_nu
class TestNuShellIntegration:
    """Integration tests against a real nu subprocess."""

    @pytest.fixture
    async def shell(self) -> AsyncGenerator[NuShellShell, None]:
        """Yield a NuShellShell and ensure it's closed after each test."""
        s = NuShellShell()
        yield s
        await s.close()

    async def test_execute_echo(self, shell: NuShellShell) -> None:
        """Execute captures stdout for a simple print."""
        result = await shell.execute("print hello")
        assert result.stdout.strip() == "hello"
        assert result.exit_code == 0

    async def test_execute_tracks_exit_code(self, shell: NuShellShell) -> None:
        """A failing external command yields a nonzero exit code."""
        result = await shell.execute("ls /nonexistent-agentsh-dir")
        assert result.exit_code != 0

    async def test_execute_tracks_cwd(self, shell: NuShellShell) -> None:
        """cwd reflects directory changes made by cd."""
        await shell.execute("cd /tmp")
        assert shell.cwd.rstrip("/") == "/tmp" or shell.cwd == "/tmp"

    async def test_can_parse_valid_and_invalid(
        self, shell: NuShellShell
    ) -> None:
        """can_parse distinguishes valid from invalid Nushell syntax."""
        assert await shell.can_parse("1 + 1") is True
        assert await shell.can_parse("{{{ not nu") is False
