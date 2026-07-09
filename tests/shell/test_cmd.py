"""Tests for the Windows CMD shell plugin."""

import os
import stat
import subprocess
import sys
from collections.abc import AsyncGenerator
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from agentsh.shell.plugin.cmd import (
    _SENTINEL,
    CmdShell,
    _complete_from_path,
    _expand_prompt,
    _parse_sentinel,
)


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
