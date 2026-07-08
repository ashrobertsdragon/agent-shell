"""Tests for the PowerShell shell plugin."""

import shutil
from collections.abc import AsyncGenerator
from pathlib import Path

import pytest

from agentsh.shell.plugin.powershell import (
    _SENTINEL,
    PowerShellShell,
    _parse_sentinel,
    _ps_quote,
    _psreadline_history_path,
    _resolve_powershell,
    _wrap_command,
)


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
    assert f'"{marker}:$__ec' in wrapped


async def test_history_round_trip(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """append_history then history round-trips without a subprocess."""
    hist = tmp_path / "sub" / "hist.txt"
    monkeypatch.setattr(
        "agentsh.shell.plugin.powershell._psreadline_history_path",
        lambda: hist,
    )
    shell = PowerShellShell()
    await shell.append_history("Get-ChildItem")
    await shell.append_history("Get-Location")
    assert await shell.history() == ["Get-ChildItem", "Get-Location"]
    assert await shell.history(limit=1) == ["Get-Location"]


async def test_history_missing_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """history returns an empty list when the file does not exist."""
    monkeypatch.setattr(
        "agentsh.shell.plugin.powershell._psreadline_history_path",
        lambda: tmp_path / "missing.txt",
    )
    shell = PowerShellShell()
    assert await shell.history() == []


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
