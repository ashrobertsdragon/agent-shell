"""Tests for the PowerShell shell plugin."""

import asyncio
import re
import shutil
import stat
import subprocess
import sys
import threading
import warnings
from collections.abc import AsyncGenerator, Callable
from pathlib import Path
from types import SimpleNamespace

import pytest

from agentsh.shell.plugin import powershell as powershell_module
from agentsh.shell.plugin.powershell import (
    _SENTINEL,
    PowerShellShell,
    _parse_sentinel,
    _ps_quote,
    _psreadline_history_path,
    _resolve_powershell,
    _strip_leading_ansi,
    _wrap_command,
)


def test_strip_leading_ansi_removes_decckm_noise() -> None:
    """A leading DECCKM set/reset pair (pwsh's own noise) is stripped."""
    assert _strip_leading_ansi("\x1b[?1h\x1b[?1lhello\n") == "hello\n"


def test_strip_leading_ansi_removes_repeated_pairs() -> None:
    """Multiple consecutive noise pairs (observed on real pwsh) are all stripped."""
    noisy = "\x1b[?1h\x1b[?1l\x1b[?1h\x1b[?1lhello\n"
    assert _strip_leading_ansi(noisy) == "hello\n"


def test_strip_leading_ansi_preserves_real_leading_color_code() -> None:
    """A real leading SGR/color code after the noise is left untouched.

    Regression guard: an earlier, broader regex matched any leading CSI
    sequence and would also eat a legitimate color code immediately
    following the noise (e.g. `git -c color.ui=always log --oneline`
    output), corrupting real command output rather than only stripping
    pwsh's own DECCKM noise.
    """
    noisy = "\x1b[?1h\x1b[?1l\x1b[33m53cb909\x1b[m init\n"
    assert _strip_leading_ansi(noisy) == "\x1b[33m53cb909\x1b[m init\n"


def test_strip_leading_ansi_passthrough_without_noise() -> None:
    """A line with no leading noise is returned unchanged."""
    assert _strip_leading_ansi("hello\n") == "hello\n"


class _FakeClock:
    """Stand-in for the time module yielding preset monotonic ticks."""

    def __init__(self, *ticks: float) -> None:
        self._ticks = iter(ticks)

    def monotonic(self) -> float:
        """Return the next preset tick."""
        return next(self._ticks)


class _FakeStdin:
    """Synthesizes a pwsh-shaped stdout reply for each write.

    execute() writes the base64-wrapped command plus the sentinel-print
    statement in a single call, then reads stdout until the sentinel
    line appears. This fake extracts the per-call marker from that
    write and feeds a canned reply straight into the paired
    StreamReader, so execute()'s wrapping and sentinel-parsing logic
    can be exercised without a real pwsh subprocess.
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


def test_resolve_prefers_pwsh(monkeypatch: pytest.MonkeyPatch) -> None:
    """pwsh wins over powershell when both are on PATH."""
    paths = {"pwsh": "/usr/bin/pwsh", "powershell": "/usr/bin/powershell"}
    monkeypatch.setattr(shutil, "which", lambda name: paths.get(name))
    assert _resolve_powershell() == "/usr/bin/pwsh"


def test_resolve_falls_back_to_powershell(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """powershell.exe is used when pwsh is absent."""
    paths = {"powershell": "C:\\ps\\powershell.exe"}
    monkeypatch.setattr(shutil, "which", lambda name: paths.get(name))
    assert _resolve_powershell() == "C:\\ps\\powershell.exe"


def test_resolve_raises_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """A RuntimeError is raised when no PowerShell executable exists."""
    monkeypatch.setattr(shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError):
        _resolve_powershell()


def test_history_path_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    """The Windows branch uses APPDATA."""
    monkeypatch.setenv("APPDATA", "C:\\Users\\x\\AppData\\Roaming")
    path = _psreadline_history_path("nt")
    expected = (
        Path("C:\\Users\\x\\AppData\\Roaming")
        / "Microsoft"
        / "Windows"
        / "PowerShell"
        / "PSReadLine"
        / "ConsoleHost_history.txt"
    )
    assert path == expected


def test_history_path_linux_xdg(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The POSIX branch honours XDG_DATA_HOME."""
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    path = _psreadline_history_path("posix")
    expected = (
        tmp_path / "powershell" / "PSReadLine" / "ConsoleHost_history.txt"
    )
    assert path == expected


def test_history_path_linux_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """The POSIX branch defaults to ~/.local/share."""
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    path = _psreadline_history_path("posix")
    expected = (
        Path.home()
        / ".local"
        / "share"
        / "powershell"
        / "PSReadLine"
        / "ConsoleHost_history.txt"
    )
    assert path == expected


def test_parse_sentinel_windows_path() -> None:
    """Drive-letter colons in cwd survive because cwd is the last field."""
    marker = f"{_SENTINEL}_nonce"
    parsed = _parse_sentinel(f"{marker}:0:C:\\Users\\x\r\n", marker)
    assert parsed == (0, "C:\\Users\\x")


def test_parse_sentinel_nonzero_code() -> None:
    """Nonzero exit codes are parsed as ints."""
    marker = f"{_SENTINEL}_nonce"
    parsed = _parse_sentinel(f"{marker}:42:/tmp\n", marker)
    assert parsed == (42, "/tmp")


def test_parse_sentinel_rejects_lookalike_without_matching_marker() -> None:
    """A line whose marker nonce differs is not treated as a match.

    This is the regression case for sentinel spoofing: command output
    that merely starts with the base sentinel string, but carries a
    different (or no) nonce, must not desync the next command.
    """
    marker = f"{_SENTINEL}_nonce-a"
    spoofed = f"{_SENTINEL}_nonce-b:0:/tmp\n"
    assert _parse_sentinel(spoofed, marker) is None


def test_parse_sentinel_rejects_malformed_line() -> None:
    """A line missing the code/cwd fields returns None instead of raising."""
    marker = f"{_SENTINEL}_nonce"
    assert _parse_sentinel(f"{marker}:not-an-int:/tmp\n", marker) is None
    assert _parse_sentinel("unrelated output\n", marker) is None


def test_ps_quote_escapes_single_quotes() -> None:
    """Embedded single quotes are doubled inside the quoted literal."""
    assert _ps_quote("it's") == "'it''s'"
    assert _ps_quote("plain") == "'plain'"


def test_wrap_command_quotes_stderr_path_with_single_quote() -> None:
    """A stderr path containing a quote is embedded as a safe literal."""
    wrapped = _wrap_command("echo hi", "/home/o'connor/stderr.txt")
    assert "'/home/o''connor/stderr.txt'" in wrapped
    assert _SENTINEL in wrapped


def test_wrap_command_embeds_supplied_marker() -> None:
    """The per-call marker (sentinel plus nonce), not the bare sentinel, is emitted."""
    marker = f"{_SENTINEL}:some-nonce"
    wrapped = _wrap_command("echo hi", "/tmp/stderr.txt", marker)
    assert f'"{marker}:${{__ec}}' in wrapped


def _patch_default_history_path(
    monkeypatch: pytest.MonkeyPatch, path: Path
) -> None:
    """Point agentsh's own PowerShell history file at path."""
    monkeypatch.setattr(
        powershell_module, "_default_history_path", lambda: path
    )


async def test_history_round_trip(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """append_history then history round-trips without a subprocess."""
    hist = tmp_path / "sub" / "hist.txt"
    _patch_default_history_path(monkeypatch, hist)
    shell = PowerShellShell()
    await shell.append_history("Get-ChildItem")
    await shell.append_history("Get-Location")
    assert await shell.history() == ["Get-ChildItem", "Get-Location"]
    assert await shell.history(limit=1) == ["Get-Location"]


async def test_history_missing_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """history returns an empty list when the file does not exist."""
    _patch_default_history_path(monkeypatch, tmp_path / "missing.txt")
    shell = PowerShellShell()
    assert await shell.history() == []


async def test_append_history_writes_own_secure_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """append_history writes agentsh's own file at mode 0o600."""
    own_file = tmp_path / "powershell_history"
    _patch_default_history_path(monkeypatch, own_file)
    shell = PowerShellShell()
    await shell.append_history("Get-ChildItem")
    assert own_file.read_text() == "Get-ChildItem\n"
    if sys.platform != "win32":
        assert stat.S_IMODE(own_file.stat().st_mode) == 0o600


async def test_append_history_does_not_write_psreadline_by_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """By default, append_history never touches PSReadLine's history."""
    _patch_default_history_path(monkeypatch, tmp_path / "own_history")
    psreadline_file = tmp_path / "ConsoleHost_history.txt"
    monkeypatch.setattr(
        powershell_module, "_psreadline_history_path", lambda: psreadline_file
    )
    shell = PowerShellShell()
    await shell.append_history("Get-ChildItem")
    assert not psreadline_file.exists()


async def test_append_history_mirrors_to_psreadline_when_env_enabled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """AGENTSH_MIRROR_PSREADLINE_HISTORY=1 mirrors and warns once."""
    _patch_default_history_path(monkeypatch, tmp_path / "own_history")
    psreadline_file = tmp_path / "ConsoleHost_history.txt"
    monkeypatch.setattr(
        powershell_module, "_psreadline_history_path", lambda: psreadline_file
    )
    monkeypatch.setenv("AGENTSH_MIRROR_PSREADLINE_HISTORY", "1")
    shell = PowerShellShell()

    with pytest.warns(UserWarning, match="AGENTSH_MIRROR_PSREADLINE_HISTORY"):
        await shell.append_history("Get-ChildItem")

    assert "Get-ChildItem" in psreadline_file.read_text()

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        await shell.append_history("Get-Location")
    assert psreadline_file.read_text().count("Get-Location") == 1


async def test_execute_full_round_trip_with_stderr() -> None:
    """execute wraps the command, parses the sentinel, and reads stderr.

    Drives the real execute() implementation against a mocked pwsh
    subprocess so the base64-wrapping, sentinel-parsing, and
    stderr-capture logic is validated on any platform, not only on
    real Windows or a machine with pwsh installed.
    """

    def respond(wrapped: str, marker: str) -> str:
        stderr_match = re.search(r"2>>'([^']+)'", wrapped)
        assert stderr_match is not None
        Path(stderr_match.group(1)).write_text("oops\n")
        return f"hello\n{marker}:0:/home/x\n"

    shell = PowerShellShell()
    shell._process = _FakeProcess(respond)  # type: ignore[assignment]
    result = await shell.execute("Write-Output hello")
    assert result.stdout == "hello\n"
    assert result.stderr == "oops\n"
    assert result.exit_code == 0
    assert result.cwd == "/home/x"
    assert shell.cwd == "/home/x"


async def test_execute_nonzero_exit_code_round_trips() -> None:
    """A nonzero $LASTEXITCODE from the mocked subprocess is preserved."""

    def respond(wrapped: str, marker: str) -> str:
        return f"{marker}:3:/home/x\n"

    shell = PowerShellShell()
    shell._process = _FakeProcess(respond)  # type: ignore[assignment]
    result = await shell.execute("exit 3")
    assert result.exit_code == 3


async def test_execute_duration_unit_consistent_on_child_process_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The ChildProcessError branch reports duration in milliseconds."""
    shell = PowerShellShell()
    shell._process = SimpleNamespace(  # type: ignore[assignment]
        stdin=None, returncode=None
    )
    monkeypatch.setattr(powershell_module, "time", _FakeClock(0.0, 0.5))
    result = await shell.execute("Get-ChildItem")
    assert result.duration_ms == 500.0
    assert result.exit_code == 1


async def test_env_parses_name_value_pairs() -> None:
    """env() splits `Name=Value` lines from the mocked Get-ChildItem env:."""

    def respond(wrapped: str, marker: str) -> str:
        body = "PATH=/usr/bin\nHOME=/home/x\n"
        return f"{body}{marker}:0:/home/x\n"

    shell = PowerShellShell()
    shell._process = _FakeProcess(respond)  # type: ignore[assignment]
    env = await shell.env()
    assert env == {"PATH": "/usr/bin", "HOME": "/home/x"}


async def test_complete_returns_completion_matches() -> None:
    """complete() returns each line of the mocked CommandCompletion output."""

    def respond(wrapped: str, marker: str) -> str:
        body = "Get-ChildItem\nGet-Command\n"
        return f"{body}{marker}:0:/home/x\n"

    shell = PowerShellShell()
    shell._process = _FakeProcess(respond)  # type: ignore[assignment]
    matches = await shell.complete("Get-Ch")
    assert matches == ["Get-ChildItem", "Get-Command"]


def _fake_completed_process(
    returncode: int,
) -> subprocess.CompletedProcess[bytes]:
    """Build a minimal CompletedProcess carrying only a returncode."""
    return subprocess.CompletedProcess(args=[], returncode=returncode)


async def test_can_parse_true_when_parser_accepts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """can_parse is True when the mocked parser subprocess exits zero."""
    monkeypatch.setattr(
        powershell_module, "_resolve_powershell", lambda: "pwsh"
    )
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: _fake_completed_process(0)
    )
    shell = PowerShellShell()
    assert await shell.can_parse("Get-ChildItem -Force") is True


async def test_can_parse_false_when_parser_rejects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """can_parse is False when the mocked parser subprocess exits nonzero."""
    monkeypatch.setattr(
        powershell_module, "_resolve_powershell", lambda: "pwsh"
    )
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: _fake_completed_process(1)
    )
    shell = PowerShellShell()
    assert await shell.can_parse("if ($x {") is False


async def test_can_parse_false_when_powershell_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """can_parse is False (not raising) when no PowerShell is on PATH."""

    def _raise() -> str:
        raise RuntimeError("no PowerShell executable found")

    monkeypatch.setattr(powershell_module, "_resolve_powershell", _raise)
    shell = PowerShellShell()
    assert await shell.can_parse("Get-ChildItem") is False


async def test_can_parse_false_on_subprocess_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """can_parse is False when the parser subprocess times out."""
    monkeypatch.setattr(
        powershell_module, "_resolve_powershell", lambda: "pwsh"
    )

    def _raise(
        *args: object, **kwargs: object
    ) -> subprocess.CompletedProcess[bytes]:
        raise subprocess.TimeoutExpired(cmd="pwsh", timeout=5.0)

    monkeypatch.setattr(subprocess, "run", _raise)
    shell = PowerShellShell()
    assert await shell.can_parse("Get-ChildItem") is False


async def test_history_read_runs_off_the_event_loop(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """history() reads the own history file via asyncio.to_thread, not inline."""
    _patch_default_history_path(monkeypatch, tmp_path / "hist.txt")
    shell = PowerShellShell()
    await shell.append_history("Get-ChildItem")

    main_thread = threading.current_thread()
    read_thread: threading.Thread | None = None
    original_read_last_lines = powershell_module.read_last_lines

    def _spy(path: Path, limit: int) -> list[str]:
        nonlocal read_thread
        read_thread = threading.current_thread()
        return original_read_last_lines(path, limit)

    monkeypatch.setattr(powershell_module, "read_last_lines", _spy)
    await shell.history()

    assert read_thread is not None
    assert read_thread is not main_thread


async def test_append_history_write_runs_off_the_event_loop(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """append_history's own-file write runs via asyncio.to_thread, not inline."""
    _patch_default_history_path(monkeypatch, tmp_path / "hist.txt")
    shell = PowerShellShell()
    main_thread = threading.current_thread()
    write_thread: threading.Thread | None = None
    original = powershell_module.append_secure_line

    def _spy(path: Path, line: str) -> None:
        nonlocal write_thread
        write_thread = threading.current_thread()
        original(path, line)

    monkeypatch.setattr(powershell_module, "append_secure_line", _spy)
    await shell.append_history("Get-ChildItem")

    assert write_thread is not None
    assert write_thread is not main_thread


requires_pwsh = pytest.mark.skipif(
    shutil.which("pwsh") is None, reason="pwsh not installed"
)


@requires_pwsh
class TestPowerShellIntegration:
    """Integration tests against a real pwsh subprocess."""

    @pytest.fixture
    async def shell(self) -> AsyncGenerator[PowerShellShell, None]:
        """Yield a PowerShellShell and ensure it's closed after each test."""
        s = PowerShellShell()
        yield s
        await s.close()

    async def test_execute_echo(self, shell: PowerShellShell) -> None:
        """Execute captures stdout."""
        result = await shell.execute('Write-Output "hello"')
        assert result.stdout.strip() == "hello"
        assert result.exit_code == 0

    async def test_execute_does_not_hang_on_command_stdin_mode(
        self, shell: PowerShellShell
    ) -> None:
        """A real pwsh command completes promptly and yields a clean sentinel.

        Regression test for #57: the ``$__ec:`` interpolation ParserError
        and the ``-Command -`` mode's VT100 escape-code prefix on every
        output line previously meant execute() never found a matching
        sentinel line and hung forever. asyncio.wait_for with a short
        timeout turns a regression of either bug into a fast, clean test
        failure instead of a suite that never completes; the stdout
        assertion also guards against the ANSI codes leaking into output.
        """
        result = await asyncio.wait_for(
            shell.execute('Write-Output "hello"'), timeout=10
        )
        assert result.stdout.strip() == "hello"
        assert result.exit_code == 0

    async def test_execute_multiline(self, shell: PowerShellShell) -> None:
        """Multi-line commands execute as a single unit."""
        result = await shell.execute(
            'if ($true) {\n    Write-Output "block"\n}'
        )
        assert result.stdout.strip() == "block"
        assert result.exit_code == 0

    async def test_execute_captures_stderr(
        self, shell: PowerShellShell
    ) -> None:
        """Execute captures the error stream separately."""
        result = await shell.execute('Write-Error "err"')
        assert "err" in result.stderr

    async def test_execute_cmdlet_failure(self, shell: PowerShellShell) -> None:
        """A failing cmdlet yields a nonzero exit code."""
        result = await shell.execute(
            "Get-Item /nonexistent-agentsh-path -ErrorAction Stop"
        )
        assert result.exit_code != 0

    async def test_execute_native_exit_code(
        self, shell: PowerShellShell
    ) -> None:
        """A native command's exit code is propagated."""
        result = await shell.execute("/bin/sh -c 'exit 3'")
        assert result.exit_code == 3

    async def test_execute_tracks_cwd(self, shell: PowerShellShell) -> None:
        """cwd reflects directory changes made by Set-Location."""
        await shell.execute("Set-Location /tmp")
        assert shell.cwd == "/tmp"

    async def test_env_contains_path(self, shell: PowerShellShell) -> None:
        """env returns the subprocess environment."""
        env = await shell.env()
        assert "PATH" in env

    async def test_complete_returns_matches(
        self, shell: PowerShellShell
    ) -> None:
        """complete returns completions for a cmdlet prefix."""
        matches = await shell.complete("Get-Ch")
        assert any("Get-Ch" in m for m in matches)

    async def test_can_parse_valid(self, shell: PowerShellShell) -> None:
        """can_parse returns True for valid syntax."""
        assert await shell.can_parse("Get-ChildItem -Force") is True

    async def test_can_parse_invalid(self, shell: PowerShellShell) -> None:
        """can_parse returns False for invalid syntax."""
        assert await shell.can_parse("if ($x {") is False

    async def test_render_prompt_returns_nonempty(
        self, shell: PowerShellShell
    ) -> None:
        """render_prompt returns a non-empty string."""
        prompt = await shell.render_prompt()
        assert isinstance(prompt, str)
        assert len(prompt) > 0
