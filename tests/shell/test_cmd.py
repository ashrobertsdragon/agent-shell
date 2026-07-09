"""Tests for the Windows CMD shell plugin."""

import asyncio
import os
import re
import stat
import subprocess
import sys
from collections.abc import AsyncGenerator, Callable
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agentsh.shell.plugin import cmd as cmd_module
from agentsh.shell.plugin.cmd import (
    _SENTINEL,
    CmdShell,
    _complete_from_path,
    _expand_prompt,
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
    """Synthesizes a cmd.exe-shaped stdout reply for each write.

    execute() writes the wrapped command plus the sentinel-echo line in
    a single call, then reads stdout until the sentinel line appears.
    This fake extracts the per-call marker from that write and feeds a
    canned reply straight into the paired StreamReader, so execute()'s
    wrapping and sentinel-parsing logic can be exercised without a real
    cmd.exe subprocess.
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


def test_expand_prompt_default() -> None:
    """$P$G expands to cwd followed by a greater-than sign."""
    assert _expand_prompt("$P$G", "C:\\Users\\x") == "C:\\Users\\x>"


def test_expand_prompt_lowercase_codes() -> None:
    """Prompt codes are case-insensitive."""
    assert _expand_prompt("$p$g", "C:\\") == "C:\\>"


def test_expand_prompt_special_codes() -> None:
    """$$, $_, $S and friends expand to their literals."""
    assert _expand_prompt("$$$S$G$L$B$Q$A", "/x") == "$ ><|=&"
    assert _expand_prompt("$P$_$G", "/x") == "/x\n>"


def test_expand_prompt_unknown_code_dropped() -> None:
    """Unknown $-codes are dropped, literal text is preserved."""
    assert _expand_prompt("hi$Z$G", "/x") == "hi>"


def test_parse_sentinel_drive_letter() -> None:
    """Drive-letter colons in %cd% survive because cwd is the last field."""
    marker = f"{_SENTINEL}_nonce"
    parsed = _parse_sentinel(f"{marker}:1:C:\\Program Files\r\n", marker)
    assert parsed == (1, "C:\\Program Files")


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


def test_complete_from_path(tmp_path: Path) -> None:
    """Executables matching prefix and PATHEXT are returned by stem."""
    (tmp_path / "git.exe").touch()
    (tmp_path / "gh.cmd").touch()
    (tmp_path / "GIMP.EXE").touch()
    (tmp_path / "notes.txt").touch()
    matches = _complete_from_path("gi", str(tmp_path), ".EXE;.CMD")
    assert "git" in matches
    assert "GIMP" in matches
    assert "notes" not in matches
    assert "gh" not in matches


def test_complete_from_path_missing_dir() -> None:
    """Nonexistent PATH entries are skipped without error."""
    assert _complete_from_path("x", "/nonexistent-agentsh-dir", ".EXE") == []


async def test_complete_includes_builtins(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """complete matches cmd builtins without spawning a subprocess."""
    monkeypatch.setenv("PATH", str(tmp_path))
    shell = CmdShell()
    matches = await shell.complete("di")
    assert "dir" in matches


def _make_shell(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    clink: str | None,
) -> CmdShell:
    """Build a CmdShell with history path and clink detection patched."""
    monkeypatch.setattr(
        "agentsh.shell.plugin.cmd._default_history_path",
        lambda: tmp_path / "agentsh" / "cmd_history",
    )
    monkeypatch.setattr("agentsh.shell.plugin.cmd._detect_clink", lambda: clink)
    return CmdShell()


async def test_history_round_trip_without_clink(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """append_history then history round-trips via the own file."""
    shell = _make_shell(monkeypatch, tmp_path, clink=None)
    await shell.append_history("dir")
    await shell.append_history("cd ..")
    assert await shell.history() == ["dir", "cd .."]
    assert await shell.history(limit=1) == ["cd .."]


async def test_history_missing_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """history returns an empty list when the file does not exist."""
    shell = _make_shell(monkeypatch, tmp_path, clink=None)
    assert await shell.history() == []


async def test_append_history_writes_own_secure_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """append_history writes the own history file at mode 0o600."""
    shell = _make_shell(monkeypatch, tmp_path, clink=None)
    await shell.append_history("dir")
    own_file = tmp_path / "agentsh" / "cmd_history"
    assert own_file.read_text() == "dir\n"
    if sys.platform != "win32":
        assert stat.S_IMODE(own_file.stat().st_mode) == 0o600


async def test_append_history_mirrors_to_clink(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """With clink present, appends go to the own file AND clink."""
    shell = _make_shell(monkeypatch, tmp_path, clink="/fake/clink")
    run_mock = MagicMock()
    monkeypatch.setattr(subprocess, "run", run_mock)
    await shell.append_history("dir")
    assert (tmp_path / "agentsh" / "cmd_history").read_text() == "dir\n"
    assert run_mock.call_args.args[0] == [
        "/fake/clink",
        "history",
        "add",
        "dir",
    ]


async def test_clink_failure_swallowed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A failing clink invocation does not break append_history."""
    shell = _make_shell(monkeypatch, tmp_path, clink="/fake/clink")
    monkeypatch.setattr(
        subprocess, "run", MagicMock(side_effect=OSError("boom"))
    )
    await shell.append_history("dir")
    assert (tmp_path / "agentsh" / "cmd_history").read_text() == "dir\n"


async def test_can_parse_always_true() -> None:
    """cmd has no syntax-check mode, so everything parses."""
    shell = CmdShell()
    assert await shell.can_parse(")((nonsense") is True


async def test_render_prompt_expands_prompt_var(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """render_prompt expands the PROMPT env var against tracked cwd."""
    monkeypatch.setenv("PROMPT", "$P$G")
    shell = CmdShell()
    prompt = await shell.render_prompt()
    assert prompt == f"{shell.cwd}>"


async def test_render_prompt_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """render_prompt falls back to $P$G when PROMPT is unset."""
    monkeypatch.delenv("PROMPT", raising=False)
    shell = CmdShell()
    prompt = await shell.render_prompt()
    assert prompt == f"{shell.cwd}>"


async def test_execute_full_round_trip_with_stderr() -> None:
    """execute wraps the command, parses the sentinel, and reads stderr.

    Drives the real execute() implementation against a mocked cmd.exe
    subprocess so the wrapping, sentinel-parsing, and stderr-capture
    logic is validated on any platform, not only on real Windows.
    """

    def respond(wrapped: str, marker: str) -> str:
        stderr_match = re.search(r'2>"([^"]+)"', wrapped)
        assert stderr_match is not None
        Path(stderr_match.group(1)).write_text("oops\n")
        return f"hello\r\n{marker}:0:C:\\Users\\x\r\n"

    shell = CmdShell()
    shell._process = _FakeProcess(respond)  # type: ignore[assignment]
    result = await shell.execute("echo hello")
    assert result.stdout == "hello\n"
    assert result.stderr == "oops\n"
    assert result.exit_code == 0
    assert result.cwd == "C:\\Users\\x"
    assert shell.cwd == "C:\\Users\\x"


async def test_execute_nonzero_exit_code_round_trips() -> None:
    """A nonzero %errorlevel% from the mocked subprocess is preserved."""

    def respond(wrapped: str, marker: str) -> str:
        return f"{marker}:1:C:\\Users\\x\r\n"

    shell = CmdShell()
    shell._process = _FakeProcess(respond)  # type: ignore[assignment]
    result = await shell.execute("dir C:\\nonexistent-agentsh-dir")
    assert result.exit_code == 1


async def test_execute_duration_unit_consistent_on_child_process_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The ChildProcessError branch reports duration in milliseconds."""
    shell = CmdShell()
    shell._process = SimpleNamespace(  # type: ignore[assignment]
        stdin=None, returncode=None
    )
    monkeypatch.setattr(cmd_module, "time", _FakeClock(0.0, 0.5))
    result = await shell.execute("dir")
    assert result.duration_ms == 500.0
    assert result.exit_code == 1


async def test_env_parses_set_output_skips_hidden_vars() -> None:
    """env() skips cmd's hidden `=C:` per-drive vars but keeps the rest."""

    def respond(wrapped: str, marker: str) -> str:
        body = "COMSPEC=C:\\Windows\\system32\\cmd.exe\r\n=C:=C:\\Users\\x\r\n"
        return f"{body}{marker}:0:C:\\Users\\x\r\n"

    shell = CmdShell()
    shell._process = _FakeProcess(respond)  # type: ignore[assignment]
    env = await shell.env()
    assert env["COMSPEC"] == "C:\\Windows\\system32\\cmd.exe"
    assert not any(k.startswith("=") for k in env)


requires_cmd = pytest.mark.skipif(
    os.name != "nt", reason="cmd.exe requires Windows"
)


@requires_cmd
class TestCmdIntegration:
    """Integration tests against a real cmd.exe subprocess."""

    @pytest.fixture
    async def shell(self) -> AsyncGenerator[CmdShell, None]:
        """Yield a CmdShell and ensure it's closed after each test."""
        s = CmdShell()
        yield s
        await s.close()

    async def test_execute_echo(self, shell: CmdShell) -> None:
        """Execute captures stdout."""
        result = await shell.execute("echo hello")
        assert result.stdout.strip() == "hello"
        assert result.exit_code == 0

    async def test_execute_captures_stderr(self, shell: CmdShell) -> None:
        """Execute captures stderr separately."""
        result = await shell.execute("echo err 1>&2")
        assert "err" in result.stderr

    async def test_execute_tracks_exit_code(self, shell: CmdShell) -> None:
        """Execute returns the last exit code."""
        result = await shell.execute("cd C:\\nonexistent-agentsh-dir")
        assert result.exit_code != 0

    async def test_execute_tracks_cwd(self, shell: CmdShell) -> None:
        """cwd reflects directory changes made by cd."""
        await shell.execute("cd C:\\")
        assert shell.cwd.rstrip("\\") == "C:"

    async def test_env_contains_comspec(self, shell: CmdShell) -> None:
        """env returns the subprocess environment via set."""
        env = await shell.env()
        assert "COMSPEC" in {k.upper() for k in env}
