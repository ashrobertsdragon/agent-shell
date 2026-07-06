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


async def test_bash_render_prompt_kills_subprocess_on_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A timed-out prompt subprocess is killed, not leaked."""
    spawn = _SpawnRecorder()
    monkeypatch.setattr(bash_module.asyncio, "create_subprocess_exec", spawn)
    monkeypatch.setattr(bash_module, "_PROMPT_TIMEOUT", 0.01, raising=False)
    shell = BashShell()
    prompt = await shell.render_prompt()
    assert prompt.endswith("$ ")
    assert spawn.proc.killed is True
    assert spawn.proc.returncode is not None


async def test_bash_render_prompt_redirects_stdin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The prompt subprocess must not inherit the REPL's stdin."""
    spawn = _SpawnRecorder()
    monkeypatch.setattr(bash_module.asyncio, "create_subprocess_exec", spawn)
    monkeypatch.setattr(bash_module, "_PROMPT_TIMEOUT", 0.01, raising=False)
    shell = BashShell()
    await shell.render_prompt()
    assert spawn.kwargs.get("stdin") == asyncio.subprocess.DEVNULL


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
