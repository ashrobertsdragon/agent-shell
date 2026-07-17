"""Integration tests for BashShell against a real bash subprocess."""

import shlex
import stat
import threading
import warnings
from collections.abc import AsyncGenerator
from pathlib import Path
from types import SimpleNamespace

import pytest

from agentsh.limits import MAX_OUTPUT_BYTES, truncation_marker
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
async def shell(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> AsyncGenerator[BashShell, None]:
    """Yield a BashShell whose own history file lives under tmp_path.

    Patching _default_history_path here (rather than only in the tests
    that exercise history directly) keeps every test using this fixture
    from ever touching the real ~/.config/agentsh/bash_history.
    """
    monkeypatch.setattr(
        bash_module, "_default_history_path", lambda: tmp_path / "bash_history"
    )
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


async def test_backend_is_interactive(shell: BashShell) -> None:
    """The persistent bash runs interactively so it sources rc files.

    ``$-`` contains ``i`` only for an interactive shell; this is what
    makes the user's aliases, functions and prompt available.
    """
    result = await shell.execute("[[ $- == *i* ]] && echo INTERACTIVE")
    assert result.stdout.strip() == "INTERACTIVE"
    assert result.exit_code == 0


async def test_history_expansion_disabled(shell: BashShell) -> None:
    """A literal ``!`` is not history-expanded despite interactive mode."""
    result = await shell.execute("echo 'a!b'")
    assert result.stdout.strip() == "a!b"
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


async def test_append_history_does_not_write_histfile_by_default(
    shell: BashShell, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """By default, append_history never touches $HISTFILE."""
    histfile = tmp_path / ".bash_history"
    monkeypatch.setenv("HISTFILE", str(histfile))
    await shell.append_history("ls -la")
    assert not histfile.exists()


async def test_append_history_writes_own_secure_file(
    shell: BashShell, tmp_path: Path
) -> None:
    """append_history writes to agentsh's own history file at mode 0o600."""
    await shell.append_history("ls -la")
    own_file = tmp_path / "bash_history"
    assert own_file.read_text() == "ls -la\n"
    assert stat.S_IMODE(own_file.stat().st_mode) == 0o600


async def test_history_round_trips_through_own_file(
    shell: BashShell,
) -> None:
    """history() reads back what append_history wrote to the own file."""
    await shell.append_history("echo one")
    await shell.append_history("echo two")
    assert await shell.history() == ["echo one", "echo two"]
    assert await shell.history(limit=1) == ["echo two"]


async def test_append_history_mirrors_to_histfile_when_env_enabled(
    shell: BashShell, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AGENTSH_MIRROR_HISTFILE=1 mirrors into $HISTFILE and warns once."""
    histfile = tmp_path / ".bash_history"
    monkeypatch.setenv("HISTFILE", str(histfile))
    monkeypatch.setenv("AGENTSH_MIRROR_HISTFILE", "1")

    with pytest.warns(UserWarning, match="AGENTSH_MIRROR_HISTFILE"):
        await shell.append_history("ls -la")

    assert "ls -la" in histfile.read_text()

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        await shell.append_history("echo again")
    assert histfile.read_text().count("echo again") == 1


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


async def test_execute_single_oversized_line_does_not_crash(
    shell: BashShell,
) -> None:
    """A line beyond asyncio's internal readline limit is truncated, not fatal.

    asyncio.StreamReader.readline() raises ValueError for a single line
    longer than its internal buffer; without handling that, this would
    propagate out of execute() uncaught.
    """
    result = await shell.execute(
        "python3 -c \"import sys; sys.stdout.write('x' * 70000)\""
    )
    assert result.exit_code == 0
    assert "output truncated" in result.stdout


async def test_execute_recovers_after_oversized_line(
    shell: BashShell,
) -> None:
    """The sentinel protocol stays in sync after an oversized line.

    A truncated/crashed read must not desync the shell such that the
    next command reads garbage or hangs.
    """
    await shell.execute(
        "python3 -c \"import sys; sys.stdout.write('x' * 70000)\""
    )
    result = await shell.execute("echo after")
    assert result.stdout.strip() == "after"
    assert result.exit_code == 0


async def test_execute_caps_output_over_one_megabyte(
    shell: BashShell,
) -> None:
    """Output beyond MAX_OUTPUT_BYTES is truncated with a marker.

    A command emitting several megabytes of ordinary line-based output
    must not be buffered whole into memory or the returned stdout.
    """
    result = await shell.execute(
        'python3 -c "'
        "import sys\n"
        "for _ in range(2000): sys.stdout.write('y' * 1000 + chr(10))"
        '"'
    )
    assert result.exit_code == 0
    assert len(result.stdout.encode()) <= MAX_OUTPUT_BYTES + len(
        truncation_marker(MAX_OUTPUT_BYTES).encode()
    )
    assert truncation_marker(MAX_OUTPUT_BYTES) in result.stdout


async def test_execute_recovers_after_large_output(shell: BashShell) -> None:
    """The shell stays usable after a command truncated for size."""
    await shell.execute(
        'python3 -c "'
        "import sys\n"
        "for _ in range(2000): sys.stdout.write('y' * 1000 + chr(10))"
        '"'
    )
    result = await shell.execute("echo after")
    assert result.stdout.strip() == "after"
    assert result.exit_code == 0


async def test_execute_stderr_io_runs_off_the_event_loop(
    shell: BashShell, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Creating, reading, and deleting the stderr scratch file must not run
    on the event loop thread, or a single command stalls every other
    coroutine waiting on the loop (issue #22).
    """
    main_thread = threading.current_thread()
    create_thread: threading.Thread | None = None
    read_thread: threading.Thread | None = None
    discard_thread: threading.Thread | None = None

    original_create = bash_module.create_stderr_tempfile
    original_read = bash_module.read_capped_text
    original_discard = bash_module.discard_stderr_tempfile

    def _spy_create() -> str:
        nonlocal create_thread
        create_thread = threading.current_thread()
        return original_create()

    def _spy_read(path: str) -> str:
        nonlocal read_thread
        read_thread = threading.current_thread()
        return original_read(path)

    def _spy_discard(path: str) -> None:
        nonlocal discard_thread
        discard_thread = threading.current_thread()
        original_discard(path)

    monkeypatch.setattr(bash_module, "create_stderr_tempfile", _spy_create)
    monkeypatch.setattr(bash_module, "read_capped_text", _spy_read)
    monkeypatch.setattr(bash_module, "discard_stderr_tempfile", _spy_discard)

    result = await shell.execute("echo hello")

    assert result.stdout.strip() == "hello"
    assert create_thread is not None and create_thread is not main_thread
    assert read_thread is not None and read_thread is not main_thread
    assert discard_thread is not None and discard_thread is not main_thread


async def test_history_read_runs_off_the_event_loop(
    shell: BashShell, monkeypatch: pytest.MonkeyPatch
) -> None:
    """history() reads the history file via asyncio.to_thread, not inline."""
    main_thread = threading.current_thread()
    read_thread: threading.Thread | None = None
    original_read_last_lines = bash_module.read_last_lines

    def _spy(path: Path, limit: int) -> list[str]:
        nonlocal read_thread
        read_thread = threading.current_thread()
        return original_read_last_lines(path, limit)

    await shell.append_history("echo one")
    monkeypatch.setattr(bash_module, "read_last_lines", _spy)
    await shell.history()

    assert read_thread is not None
    assert read_thread is not main_thread


async def test_append_history_write_runs_off_the_event_loop(
    shell: BashShell, monkeypatch: pytest.MonkeyPatch
) -> None:
    """append_history's own-file write runs via asyncio.to_thread, not inline."""
    main_thread = threading.current_thread()
    write_thread: threading.Thread | None = None
    original = bash_module.append_secure_line

    def _spy(path: Path, line: str) -> None:
        nonlocal write_thread
        write_thread = threading.current_thread()
        original(path, line)

    monkeypatch.setattr(bash_module, "append_secure_line", _spy)
    await shell.append_history("echo tracked")

    assert write_thread is not None
    assert write_thread is not main_thread
