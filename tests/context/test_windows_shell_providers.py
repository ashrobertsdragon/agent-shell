"""Regression tests for context providers run against Windows shell backends.

None of src/agentsh/context/providers/*.py branches on ``os.name`` — the
one Windows-specific branch in the codebase,
``agentsh.shell.plugin.powershell._psreadline_history_path``, is already
covered by ``tests/shell/test_powershell.py::test_history_path_windows``.

Issue #19 fixed a real portability bug: the git, docker, python, and
kubernetes providers used to unconditionally embed POSIX-only shell
syntax in the commands they send to ``shell.execute`` — ``2>/dev/null``
stderr redirection and, for docker/kubernetes, single-quoted values.
``/dev/null`` is not a path that exists on native Windows shells (cmd.exe
or PowerShell); there is no such device, so redirecting to it does not
silently discard stderr the way it does on POSIX. On a real cmd.exe, for
example, ``2>/dev/null`` fails to open the target path and the whole
command line errors out with a nonzero exit code — *even when the
underlying tool (git/docker/kubectl/python) is installed and would
otherwise succeed*. Single quotes are similarly non-portable: cmd.exe
does not treat them as a quoting character, so they pass through to the
invoked program literally.

Because ``ContextBuilder._safe_collect`` (see ``context/builder.py``)
swallows all exceptions and non-zero exits, the old bug was silent: the
git/docker/kubectl/python context fragment simply never appeared on
Windows, with no error surfaced anywhere.

This module has two tiers of coverage, matching the pattern used for
other Windows-only behavior in this repo (see ``requires_cmd`` /
``requires_pwsh`` in ``tests/shell/test_cmd.py`` and
``tests/shell/test_powershell.py``):

- Command-construction tests that run on any platform: a fake Shell
  models how CmdShell/PowerShellShell actually respond to POSIX-only
  syntax (nonzero exit, English "path not found" stderr) so a provider
  that regresses to embedding ``2>/dev/null`` or single quotes is
  caught everywhere, not only on a Windows CI runner.
- Real integration tests against an actual ``CmdShell`` or
  ``PowerShellShell`` subprocess, skipped unless that backend can
  actually run (a Windows host for cmd.exe; ``pwsh``/``powershell`` on
  PATH for PowerShell, which is cross-platform).
"""

import os
import shutil
from collections.abc import AsyncGenerator
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from agentsh.context.providers.docker import DockerProvider
from agentsh.context.providers.git import GitProvider
from agentsh.context.providers.kubernetes import KubernetesProvider
from agentsh.context.providers.python import PythonProvider
from agentsh.models import CommandResult
from agentsh.shell.plugin.cmd import CmdShell
from agentsh.shell.plugin.powershell import PowerShellShell


def _windows_like_shell() -> MagicMock:
    """A Shell double simulating cmd.exe/PowerShell's handling of commands.

    Neither cmd.exe nor PowerShell has a `/dev/null` device, and neither
    treats a single quote as a quoting character the way POSIX shells
    do. Any command containing either would fail (or send garbled
    arguments) on a real Windows shell, so this double fails the test
    outright if a provider regresses to emitting either — the fixed
    providers should never do so, letting the commands below "succeed"
    with plausible Windows-shaped output.
    """
    responses = {
        "git rev-parse --abbrev-ref HEAD": CommandResult(
            stdout="main\n",
            stderr="",
            exit_code=0,
            duration_ms=1.0,
            cwd="C:\\Users\\agentsh",
        ),
        "git status --short": CommandResult(
            stdout="",
            stderr="",
            exit_code=0,
            duration_ms=1.0,
            cwd="C:\\Users\\agentsh",
        ),
        "python3 --version": CommandResult(
            stdout="",
            stderr="'python3' is not recognized as an internal or "
            "external command.\n",
            exit_code=1,
            duration_ms=1.0,
            cwd="C:\\Users\\agentsh",
        ),
        "python --version": CommandResult(
            stdout="Python 3.12.0\n",
            stderr="",
            exit_code=0,
            duration_ms=1.0,
            cwd="C:\\Users\\agentsh",
        ),
        'docker ps --format "{{.Names}}\t{{.Image}}\t{{.Status}}"': (
            CommandResult(
                stdout="web\tnginx\tUp\n",
                stderr="",
                exit_code=0,
                duration_ms=1.0,
                cwd="C:\\Users\\agentsh",
            )
        ),
        "kubectl config current-context": CommandResult(
            stdout="minikube\n",
            stderr="",
            exit_code=0,
            duration_ms=1.0,
            cwd="C:\\Users\\agentsh",
        ),
        'kubectl config view --minify -o jsonpath="{..namespace}"': (
            CommandResult(
                stdout="default\n",
                stderr="",
                exit_code=0,
                duration_ms=1.0,
                cwd="C:\\Users\\agentsh",
            )
        ),
    }

    async def execute(command: str) -> CommandResult:
        assert "/dev/null" not in command, (
            f"provider issued a non-portable redirected command: {command!r}"
        )
        assert "'" not in command, (
            f"provider issued a non-portable single-quoted command: {command!r}"
        )
        try:
            return responses[command]
        except KeyError:
            raise AssertionError(
                f"unexpected command sent to Windows-like shell: {command!r}"
            ) from None

    shell = MagicMock()
    shell.execute = execute
    shell.cwd = "C:\\Users\\agentsh"
    return shell


async def test_git_provider_returns_fragment_on_windows_shell() -> None:
    """GitProvider succeeds on a Windows shell now that redirection is gone."""
    result = await GitProvider().collect(_windows_like_shell())
    assert result is not None
    assert result.payload["branch"] == "main"


async def test_docker_provider_returns_fragment_on_windows_shell() -> None:
    """DockerProvider succeeds on a Windows shell now that quoting is fixed."""
    result = await DockerProvider().collect(_windows_like_shell())
    assert result is not None
    assert result.payload["containers"] == [
        {"name": "web", "image": "nginx", "status": "Up"}
    ]


async def test_python_provider_returns_fragment_on_windows_shell() -> None:
    """PythonProvider falls back to `python` when `python3` is missing."""
    result = await PythonProvider().collect(_windows_like_shell())
    assert result is not None
    assert result.payload["python_version"] == "3.12.0"


async def test_kubernetes_provider_returns_fragment_on_windows_shell() -> None:
    """KubernetesProvider succeeds on a Windows shell now that quoting is fixed."""
    result = await KubernetesProvider().collect(_windows_like_shell())
    assert result is not None
    assert result.payload["context"] == "minikube"
    assert result.payload["namespace"] == "default"


requires_cmd = pytest.mark.skipif(
    os.name != "nt", reason="cmd.exe requires Windows"
)
requires_pwsh = pytest.mark.skipif(
    shutil.which("pwsh") is None and shutil.which("powershell") is None,
    reason="pwsh/powershell not installed",
)


@requires_cmd
class TestProvidersOnCmdShell:
    """Providers driven against a real cmd.exe subprocess."""

    @pytest.fixture
    async def shell(self) -> AsyncGenerator[CmdShell, None]:
        """Yield a CmdShell and ensure it's closed after each test."""
        s = CmdShell()
        yield s
        await s.close()

    async def test_git_provider_reports_branch_in_repo(
        self, shell: CmdShell, tmp_path: Path
    ) -> None:
        """GitProvider reports the branch when cwd is a real git repo."""
        if shutil.which("git") is None:
            pytest.skip("git not installed")
        await shell.execute(f'cd /d "{tmp_path}"')
        await shell.execute("git init -b main")
        await shell.execute(
            "git -c user.name=test -c user.email=test@example.com "
            "commit --allow-empty -m init"
        )
        result = await GitProvider().collect(shell)
        assert result is not None
        assert result.payload["branch"] == "main"

    async def test_git_provider_returns_none_outside_repo(
        self, shell: CmdShell, tmp_path: Path
    ) -> None:
        """GitProvider returns None outside any git repository."""
        if shutil.which("git") is None:
            pytest.skip("git not installed")
        await shell.execute(f'cd /d "{tmp_path}"')
        result = await GitProvider().collect(shell)
        assert result is None

    async def test_python_provider_reports_a_version(
        self, shell: CmdShell
    ) -> None:
        """PythonProvider resolves either `python3` or `python` on PATH."""
        if shutil.which("python3") is None and shutil.which("python") is None:
            pytest.skip("no python interpreter on PATH")
        result = await PythonProvider().collect(shell)
        assert result is not None
        assert result.payload["python_version"]

    async def test_docker_provider_does_not_raise(
        self, shell: CmdShell
    ) -> None:
        """DockerProvider degrades to None rather than raising when absent."""
        result = await DockerProvider().collect(shell)
        assert result is None or "containers" in result.payload

    async def test_kubernetes_provider_does_not_raise(
        self, shell: CmdShell
    ) -> None:
        """KubernetesProvider degrades to None rather than raising when absent."""
        result = await KubernetesProvider().collect(shell)
        assert result is None or "context" in result.payload


@requires_pwsh
class TestProvidersOnPowerShellShell:
    """Providers driven against a real PowerShell subprocess."""

    @pytest.fixture
    async def shell(self) -> AsyncGenerator[PowerShellShell, None]:
        """Yield a PowerShellShell and ensure it's closed after each test."""
        s = PowerShellShell()
        yield s
        await s.close()

    async def test_git_provider_reports_branch_in_repo(
        self, shell: PowerShellShell, tmp_path: Path
    ) -> None:
        """GitProvider reports the branch when cwd is a real git repo."""
        if shutil.which("git") is None:
            pytest.skip("git not installed")
        await shell.execute(f"Set-Location '{tmp_path}'")
        await shell.execute("git init -b main")
        await shell.execute(
            "git -c user.name=test -c user.email=test@example.com "
            "commit --allow-empty -m init"
        )
        result = await GitProvider().collect(shell)
        assert result is not None
        assert result.payload["branch"] == "main"

    async def test_git_provider_returns_none_outside_repo(
        self, shell: PowerShellShell, tmp_path: Path
    ) -> None:
        """GitProvider returns None outside any git repository."""
        if shutil.which("git") is None:
            pytest.skip("git not installed")
        await shell.execute(f"Set-Location '{tmp_path}'")
        result = await GitProvider().collect(shell)
        assert result is None

    async def test_python_provider_reports_a_version(
        self, shell: PowerShellShell
    ) -> None:
        """PythonProvider resolves either `python3` or `python` on PATH."""
        if shutil.which("python3") is None and shutil.which("python") is None:
            pytest.skip("no python interpreter on PATH")
        result = await PythonProvider().collect(shell)
        assert result is not None
        assert result.payload["python_version"]

    async def test_docker_provider_does_not_raise(
        self, shell: PowerShellShell
    ) -> None:
        """DockerProvider degrades to None rather than raising when absent."""
        result = await DockerProvider().collect(shell)
        assert result is None or "containers" in result.payload

    async def test_kubernetes_provider_does_not_raise(
        self, shell: PowerShellShell
    ) -> None:
        """KubernetesProvider degrades to None rather than raising when absent."""
        result = await KubernetesProvider().collect(shell)
        assert result is None or "context" in result.payload
