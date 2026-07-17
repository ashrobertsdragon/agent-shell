"""Subprocess hygiene tests for render_prompt in bash and powershell."""

import asyncio

import pytest

from agentsh.shell.plugin import bash as bash_module
from agentsh.shell.plugin import powershell as ps_module
from agentsh.shell.plugin.bash import BashShell
from agentsh.shell.plugin.powershell import PowerShellShell


class _HangingProc:
    """Fake subprocess whose communicate() never finishes."""

    def __init__(self) -> None:
        self.killed = False
        self.returncode: int | None = None

    async def communicate(self) -> tuple[bytes, bytes]:
        """Block until cancelled."""
        await asyncio.sleep(3600)
        return b"", b""

    def kill(self) -> None:
        """Record that the process was killed."""
        self.killed = True

    async def wait(self) -> int:
        """Simulate reaping the killed process."""
        self.returncode = -9
        return -9


class _SpawnRecorder:
    """Records create_subprocess_exec kwargs and returns a hanging proc."""

    def __init__(self) -> None:
        self.kwargs: dict[str, object] = {}
        self.proc = _HangingProc()

    async def __call__(self, *args: object, **kwargs: object) -> _HangingProc:
        """Stand-in for asyncio.create_subprocess_exec."""
        self.kwargs = kwargs
        return self.proc


class _FakeStdin:
    """Writable stand-in for a subprocess stdin StreamWriter."""

    def write(self, data: bytes) -> None:
        """Discard written bytes."""

    async def drain(self) -> None:
        """No-op drain."""

    def is_closing(self) -> bool:
        """Report the writer as open."""
        return False

    def close(self) -> None:
        """No-op close."""


class _HangingStdout:
    """stdout stand-in whose reads never complete, forcing a timeout."""

    async def readuntil(self, sep: bytes) -> bytes:
        """Block until cancelled."""
        await asyncio.sleep(3600)
        return b""


class _PersistentFakeProc:
    """A live-looking persistent subprocess for render_prompt tests."""

    def __init__(self, stdout: object) -> None:
        self.returncode: int | None = None
        self.stdin = _FakeStdin()
        self.stdout = stdout


def _reader(data: bytes) -> asyncio.StreamReader:
    """Return a StreamReader pre-loaded with data and EOF."""
    reader = asyncio.StreamReader()
    reader.feed_data(data)
    reader.feed_eof()
    return reader


async def test_bash_render_prompt_falls_back_on_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A prompt read that never yields a sentinel falls back to cwd$ ."""
    shell = BashShell()
    shell._process = _PersistentFakeProc(_HangingStdout())  # type: ignore[assignment]
    shell._desynced = False
    monkeypatch.setattr(bash_module, "_PROMPT_TIMEOUT", 0.01, raising=False)
    prompt = await shell.render_prompt()
    assert prompt == f"{shell.cwd}$ "


async def test_bash_render_prompt_parses_from_persistent_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """render_prompt reads PS1 from the live subprocess, strips the one
    injected trailing newline and readline non-printing markers, and
    ignores the sentinel line.
    """
    monkeypatch.setattr(bash_module, "new_marker", lambda _s: "MARK")
    shell = BashShell()
    shell._process = _PersistentFakeProc(  # type: ignore[assignment]
        _reader(b"my\x01prompt\x02\nMARK\n")
    )
    shell._desynced = False
    prompt = await shell.render_prompt()
    assert prompt == "myprompt"


async def test_powershell_render_prompt_kills_subprocess_on_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A timed-out prompt subprocess is killed, not leaked."""
    spawn = _SpawnRecorder()
    monkeypatch.setattr(ps_module, "_resolve_powershell", lambda: "pwsh")
    monkeypatch.setattr(ps_module.asyncio, "create_subprocess_exec", spawn)
    monkeypatch.setattr(ps_module, "_PROMPT_TIMEOUT", 0.01, raising=False)
    shell = PowerShellShell()
    prompt = await shell.render_prompt()
    assert prompt.endswith("> ")
    assert spawn.proc.killed is True
    assert spawn.proc.returncode is not None


async def test_powershell_render_prompt_redirects_stdin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The prompt subprocess must not inherit the REPL's stdin."""
    spawn = _SpawnRecorder()
    monkeypatch.setattr(ps_module, "_resolve_powershell", lambda: "pwsh")
    monkeypatch.setattr(ps_module.asyncio, "create_subprocess_exec", spawn)
    monkeypatch.setattr(ps_module, "_PROMPT_TIMEOUT", 0.01, raising=False)
    shell = PowerShellShell()
    await shell.render_prompt()
    assert spawn.kwargs.get("stdin") == asyncio.subprocess.DEVNULL


async def test_powershell_render_prompt_falls_back_when_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """render_prompt falls back to `PS <cwd>> ` when no PowerShell is found.

    Covers the RuntimeError branch without needing a real pwsh/powershell
    executable on the test machine.
    """

    def _raise() -> str:
        raise RuntimeError("no PowerShell executable found")

    monkeypatch.setattr(ps_module, "_resolve_powershell", _raise)
    shell = PowerShellShell()
    prompt = await shell.render_prompt()
    assert prompt == f"PS {shell.cwd}> "


async def test_powershell_render_prompt_falls_back_on_spawn_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """render_prompt falls back when spawning the prompt subprocess fails.

    Covers the OSError branch (e.g. the resolved executable vanished
    between resolution and spawn) without needing a real subprocess.
    """

    async def _raise(*args: object, **kwargs: object) -> None:
        raise OSError("no such file or directory")

    monkeypatch.setattr(ps_module, "_resolve_powershell", lambda: "pwsh")
    monkeypatch.setattr(ps_module.asyncio, "create_subprocess_exec", _raise)
    shell = PowerShellShell()
    prompt = await shell.render_prompt()
    assert prompt == f"PS {shell.cwd}> "
