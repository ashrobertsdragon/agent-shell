"""Regression tests for context providers run against a Windows shell backend.

None of src/agentsh/context/providers/*.py branches on ``os.name`` — the
one Windows-specific branch in the codebase,
``agentsh.shell.plugin.powershell._psreadline_history_path``, is already
covered by ``tests/shell/test_powershell.py::test_history_path_windows``.

What *is* untested is that the git, docker, python, and kubernetes
providers unconditionally embed POSIX-only shell syntax in the commands
they send to ``shell.execute`` — ``2>/dev/null`` stderr redirection and,
for docker, a single-quoted ``--format`` string. ``/dev/null`` is not a
path that exists on native Windows shells (cmd.exe or PowerShell); there
is no such device, so redirecting to it does not silently discard
stderr the way it does on POSIX. On a real cmd.exe, for example,
``2>/dev/null`` fails to open the target path and the whole command line
errors out with a nonzero exit code — *even when the underlying tool
(git/docker/kubectl/python) is installed and would otherwise succeed*.

These tests pin that behavior against a fake Shell modeling how
CmdShell/PowerShellShell actually respond to such a command, so the
Windows-portability gap is a visible, checked test rather than only a
comment — and so a provider that becomes Windows-aware must
deliberately update this file rather than silently regress.

Fixing the providers to be shell-portable is out of scope here (see the
task report); this documents the current gap for issue #18's "providers
on Windows shells: no test" item.
"""

from unittest.mock import MagicMock

from agentsh.context.providers.docker import DockerProvider
from agentsh.context.providers.git import GitProvider
from agentsh.context.providers.kubernetes import KubernetesProvider
from agentsh.context.providers.python import PythonProvider
from agentsh.models import CommandResult


def _windows_like_shell() -> MagicMock:
    """A Shell double simulating cmd.exe/PowerShell's handling of `/dev/null`.

    Neither cmd.exe nor PowerShell has a `/dev/null` device; redirecting
    stderr there fails to open the target path rather than discarding
    output, so the wrapping command line comes back with a nonzero exit
    code regardless of whether the invoked tool itself would have
    succeeded.
    """

    async def execute(command: str) -> CommandResult:
        assert "/dev/null" in command, (
            f"provider issued a non-POSIX-redirected command: {command!r}"
        )
        return CommandResult(
            stdout="",
            stderr="The system cannot find the path specified.\n",
            exit_code=1,
            duration_ms=1.0,
            cwd="C:\\Users\\agentsh",
        )

    shell = MagicMock()
    shell.execute = execute
    shell.cwd = "C:\\Users\\agentsh"
    return shell


async def test_git_provider_returns_none_on_windows_shell() -> None:
    """GitProvider's POSIX `2>/dev/null` breaks on a Windows shell.

    Even inside a real git repo, the hardcoded redirection means the
    provider silently reports "not a git repo" (returns None) on a
    native Windows shell.
    """
    result = await GitProvider().collect(_windows_like_shell())
    assert result is None


async def test_docker_provider_returns_none_on_windows_shell() -> None:
    """DockerProvider's POSIX redirection breaks on a Windows shell."""
    result = await DockerProvider().collect(_windows_like_shell())
    assert result is None


async def test_python_provider_returns_none_on_windows_shell() -> None:
    """PythonProvider's POSIX `2>/dev/null` breaks on a Windows shell."""
    result = await PythonProvider().collect(_windows_like_shell())
    assert result is None


async def test_kubernetes_provider_returns_none_on_windows_shell() -> None:
    """KubernetesProvider's POSIX `2>/dev/null` breaks on a Windows shell."""
    result = await KubernetesProvider().collect(_windows_like_shell())
    assert result is None
