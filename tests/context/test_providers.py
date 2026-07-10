"""Tests for context providers."""

import threading
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from agentsh.context.providers import filesystem as filesystem_module
from agentsh.context.providers.docker import DockerProvider
from agentsh.context.providers.environment import EnvironmentProvider
from agentsh.context.providers.filesystem import FilesystemProvider
from agentsh.context.providers.git import GitProvider
from agentsh.context.providers.history import HistoryProvider
from agentsh.context.providers.kubernetes import KubernetesProvider
from agentsh.context.providers.node import NodeProvider
from agentsh.context.providers.python import PythonProvider
from agentsh.models import CommandResult


@pytest.fixture
def shell() -> MagicMock:
    """Minimal shell mock."""
    return MagicMock()


async def test_git_provider_returns_fragment_in_git_repo(
    shell: MagicMock,
) -> None:
    """GitProvider returns a fragment when inside a git repo."""
    shell.execute = AsyncMock(
        side_effect=[
            CommandResult(
                stdout="main\n",
                stderr="",
                exit_code=0,
                duration_ms=1,
                cwd="/repo",
            ),
            CommandResult(
                stdout=" M file.py\n",
                stderr="",
                exit_code=0,
                duration_ms=1,
                cwd="/repo",
            ),
        ]
    )
    provider = GitProvider()
    result = await provider.collect(shell)
    assert result is not None
    assert result.payload["branch"] == "main"


async def test_git_provider_returns_none_outside_repo(shell: MagicMock) -> None:
    """GitProvider returns None when not in a git repository."""
    shell.execute = AsyncMock(
        return_value=CommandResult(
            stdout="",
            stderr="fatal: not a git repository\n",
            exit_code=128,
            duration_ms=1,
            cwd="/tmp",
        )
    )
    provider = GitProvider()
    result = await provider.collect(shell)
    assert result is None


async def test_git_provider_commands_are_shell_portable(
    shell: MagicMock,
) -> None:
    """GitProvider never embeds POSIX-only redirection in its commands.

    ``2>/dev/null`` is redundant (CommandResult already separates
    stdout/stderr/exit_code) and breaks on cmd.exe/PowerShell, which
    have no ``/dev/null`` device.
    """
    commands: list[str] = []

    async def execute(command: str) -> CommandResult:
        commands.append(command)
        return CommandResult(
            stdout="main\n", stderr="", exit_code=0, duration_ms=1, cwd="/repo"
        )

    shell.execute = execute
    await GitProvider().collect(shell)
    assert commands
    assert all("/dev/null" not in c for c in commands)


async def test_kubernetes_provider_returns_fragment_in_cluster(
    shell: MagicMock,
) -> None:
    """KubernetesProvider returns a fragment when kubectl has a context."""
    shell.execute = AsyncMock(
        side_effect=[
            CommandResult(
                stdout="minikube\n",
                stderr="",
                exit_code=0,
                duration_ms=1,
                cwd="/repo",
            ),
            CommandResult(
                stdout="default\n",
                stderr="",
                exit_code=0,
                duration_ms=1,
                cwd="/repo",
            ),
        ]
    )
    provider = KubernetesProvider()
    result = await provider.collect(shell)
    assert result is not None
    assert result.payload["context"] == "minikube"
    assert result.payload["namespace"] == "default"


async def test_kubernetes_provider_returns_none_without_kubectl(
    shell: MagicMock,
) -> None:
    """KubernetesProvider returns None when kubectl is unavailable."""
    shell.execute = AsyncMock(
        return_value=CommandResult(
            stdout="", stderr="", exit_code=1, duration_ms=1, cwd="/repo"
        )
    )
    provider = KubernetesProvider()
    result = await provider.collect(shell)
    assert result is None


async def test_kubernetes_provider_commands_are_shell_portable(
    shell: MagicMock,
) -> None:
    """KubernetesProvider never embeds POSIX-only redirection in its commands."""
    commands: list[str] = []

    async def execute(command: str) -> CommandResult:
        commands.append(command)
        return CommandResult(
            stdout="minikube\n",
            stderr="",
            exit_code=0,
            duration_ms=1,
            cwd="/repo",
        )

    shell.execute = execute
    await KubernetesProvider().collect(shell)
    assert commands
    assert all("/dev/null" not in c for c in commands)


async def test_filesystem_provider_returns_fragment(
    shell: MagicMock, tmp_path: Path
) -> None:
    """FilesystemProvider returns a fragment listing the cwd."""
    shell.cwd = str(tmp_path)
    (tmp_path / "main.py").touch()
    provider = FilesystemProvider()
    result = await provider.collect(shell)
    assert result is not None
    files = result.payload.get("files", [])
    assert isinstance(files, list)
    assert "main.py" in files


async def test_filesystem_provider_lists_off_the_event_loop(
    shell: MagicMock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The directory listing runs via asyncio.to_thread, not inline.

    A cwd with a huge listing (e.g. node_modules) must not block the
    event loop for the duration of the scan (issue #22).
    """
    shell.cwd = str(tmp_path)
    (tmp_path / "main.py").touch()

    main_thread = threading.current_thread()
    list_thread: threading.Thread | None = None
    original = filesystem_module._list_entries

    def _spy(cwd: str) -> list[str]:
        nonlocal list_thread
        list_thread = threading.current_thread()
        return original(cwd)

    monkeypatch.setattr(filesystem_module, "_list_entries", _spy)
    provider = FilesystemProvider()
    result = await provider.collect(shell)

    assert result is not None
    assert list_thread is not None
    assert list_thread is not main_thread


async def test_python_env_provider(shell: MagicMock) -> None:
    """PythonEnvProvider returns a fragment with python version info."""
    shell.execute = AsyncMock(
        side_effect=[
            CommandResult(
                stdout="Python 3.12.0\n",
                stderr="",
                exit_code=0,
                duration_ms=1,
                cwd="/repo",
            ),
            CommandResult(
                stdout="none\n",
                stderr="",
                exit_code=0,
                duration_ms=1,
                cwd="/repo",
            ),
        ]
    )
    shell.cwd = "/repo"
    provider = PythonProvider()
    result = await provider.collect(shell)
    assert result is not None
    assert result.payload.get("python_version") == "3.12.0"


async def test_python_env_provider_falls_back_to_python_when_python3_missing(
    shell: MagicMock,
) -> None:
    """PythonProvider falls back to ``python`` when ``python3`` is absent.

    Windows shells typically only have ``python`` on PATH, not
    ``python3``.
    """
    commands: list[str] = []

    async def execute(command: str) -> CommandResult:
        commands.append(command)
        if command.startswith("python3"):
            return CommandResult(
                stdout="",
                stderr="'python3' is not recognized",
                exit_code=1,
                duration_ms=1,
                cwd="C:\\Users\\agentsh",
            )
        return CommandResult(
            stdout="Python 3.12.0\n",
            stderr="",
            exit_code=0,
            duration_ms=1,
            cwd="C:\\Users\\agentsh",
        )

    shell.execute = execute
    shell.cwd = "C:\\Users\\agentsh"
    provider = PythonProvider()
    result = await provider.collect(shell)
    assert result is not None
    assert result.payload.get("python_version") == "3.12.0"
    assert commands == ["python3 --version", "python --version"]


async def test_python_env_provider_reads_version_from_stderr(
    shell: MagicMock,
) -> None:
    """PythonProvider accepts a version string printed to stderr.

    Some Python builds print ``--version`` output to stderr even on a
    zero exit code.
    """
    shell.execute = AsyncMock(
        return_value=CommandResult(
            stdout="",
            stderr="Python 2.7.18\n",
            exit_code=0,
            duration_ms=1,
            cwd="/repo",
        )
    )
    shell.cwd = "/repo"
    provider = PythonProvider()
    result = await provider.collect(shell)
    assert result is not None
    assert result.payload.get("python_version") == "2.7.18"


async def test_python_env_provider_detects_windows_venv_layout(
    shell: MagicMock, tmp_path: Path
) -> None:
    """PythonProvider recognizes a venv laid out Windows-style.

    Windows virtualenvs place the interpreter under
    ``Scripts\\python.exe`` rather than the POSIX ``bin/python``.
    """
    (tmp_path / ".venv" / "Scripts").mkdir(parents=True)
    (tmp_path / ".venv" / "Scripts" / "python.exe").touch()
    shell.execute = AsyncMock(
        return_value=CommandResult(
            stdout="Python 3.12.0\n",
            stderr="",
            exit_code=0,
            duration_ms=1,
            cwd=str(tmp_path),
        )
    )
    shell.cwd = str(tmp_path)
    provider = PythonProvider()
    result = await provider.collect(shell)
    assert result is not None
    assert result.payload.get("has_venv") is True


async def test_python_env_provider_returns_none_when_neither_present(
    shell: MagicMock,
) -> None:
    """PythonProvider returns None when neither python3 nor python is found."""
    shell.execute = AsyncMock(
        return_value=CommandResult(
            stdout="",
            stderr="command not found",
            exit_code=127,
            duration_ms=1,
            cwd="/repo",
        )
    )
    provider = PythonProvider()
    result = await provider.collect(shell)
    assert result is None


async def test_python_provider_commands_are_shell_portable(
    shell: MagicMock,
) -> None:
    """PythonProvider never embeds POSIX-only redirection in its commands."""
    commands: list[str] = []

    async def execute(command: str) -> CommandResult:
        commands.append(command)
        return CommandResult(
            stdout="Python 3.12.0\n",
            stderr="",
            exit_code=0,
            duration_ms=1,
            cwd="/repo",
        )

    shell.execute = execute
    shell.cwd = "/repo"
    await PythonProvider().collect(shell)
    assert commands
    assert all("/dev/null" not in c for c in commands)


async def test_docker_provider_returns_none_without_docker(
    shell: MagicMock,
) -> None:
    """DockerProvider returns None when docker is unavailable."""
    shell.execute = AsyncMock(
        return_value=CommandResult(
            stdout="",
            stderr="",
            exit_code=1,
            duration_ms=1,
            cwd="/repo",
        )
    )
    provider = DockerProvider()
    result = await provider.collect(shell)
    assert result is None


async def test_docker_provider_commands_are_shell_portable(
    shell: MagicMock,
) -> None:
    """DockerProvider avoids POSIX-only redirection and single-quoting.

    ``2>/dev/null`` breaks on cmd.exe/PowerShell. Single-quoted format
    strings are also non-portable: cmd.exe does not strip single quotes
    as a quoting character, so the literal quote characters would be
    passed straight through to docker.
    """
    commands: list[str] = []

    async def execute(command: str) -> CommandResult:
        commands.append(command)
        return CommandResult(
            stdout="web\tnginx\tUp\n",
            stderr="",
            exit_code=0,
            duration_ms=1,
            cwd="/repo",
        )

    shell.execute = execute
    await DockerProvider().collect(shell)
    assert commands
    assert all("/dev/null" not in c for c in commands)
    assert all("'" not in c for c in commands)


async def test_history_provider(shell: MagicMock) -> None:
    """HistoryProvider returns recent shell commands."""
    shell.history = AsyncMock(return_value=["ls", "cd /tmp", "git status"])
    provider = HistoryProvider()
    result = await provider.collect(shell)
    assert result is not None
    assert result.payload["recent"] == ["ls", "cd /tmp", "git status"]


async def test_environment_provider(shell: MagicMock) -> None:
    """EnvironmentProvider filters out sensitive env vars."""
    shell.env = AsyncMock(
        return_value={
            "HOME": "/home/user",
            "ANTHROPIC_API_KEY": "sk-secret",
            "PATH": "/usr/bin",
            "MY_SECRET": "hidden",
        }
    )
    provider = EnvironmentProvider()
    result = await provider.collect(shell)
    assert result is not None
    env = result.payload["env"]
    assert isinstance(env, dict)
    assert "HOME" in env
    assert "PATH" in env
    assert "ANTHROPIC_API_KEY" not in env
    assert "MY_SECRET" not in env


async def test_node_provider_returns_none_without_node(
    shell: MagicMock,
) -> None:
    """NodeProvider returns None when node is unavailable."""
    shell.execute = AsyncMock(
        return_value=CommandResult(
            stdout="",
            stderr="command not found",
            exit_code=127,
            duration_ms=1,
            cwd="/repo",
        )
    )
    provider = NodeProvider()
    result = await provider.collect(shell)
    assert result is None


async def test_node_provider_strips_v_prefix_from_version(
    shell: MagicMock, tmp_path: Path
) -> None:
    """NodeProvider strips the leading ``v`` from ``node --version``."""
    shell.execute = AsyncMock(
        return_value=CommandResult(
            stdout="v20.11.0\n",
            stderr="",
            exit_code=0,
            duration_ms=1,
            cwd=str(tmp_path),
        )
    )
    shell.cwd = str(tmp_path)
    provider = NodeProvider()
    result = await provider.collect(shell)
    assert result is not None
    assert result.payload.get("node_version") == "20.11.0"


async def test_node_provider_uses_stderr_version_when_stdout_empty(
    shell: MagicMock, tmp_path: Path
) -> None:
    """NodeProvider falls back to stderr when stdout is empty."""
    shell.execute = AsyncMock(
        return_value=CommandResult(
            stdout="",
            stderr="v20.11.0\n",
            exit_code=0,
            duration_ms=1,
            cwd=str(tmp_path),
        )
    )
    shell.cwd = str(tmp_path)
    provider = NodeProvider()
    result = await provider.collect(shell)
    assert result is not None
    assert result.payload.get("node_version") == "20.11.0"


async def test_node_provider_returns_fragment_without_package_json(
    shell: MagicMock, tmp_path: Path
) -> None:
    """NodeProvider still returns a fragment when there is no package.json.

    The Node version alone is useful context even without a JS project
    in the current directory, so absence of the manifest degrades to
    empty scripts/dependencies rather than None.
    """
    shell.execute = AsyncMock(
        return_value=CommandResult(
            stdout="v20.11.0\n",
            stderr="",
            exit_code=0,
            duration_ms=1,
            cwd=str(tmp_path),
        )
    )
    shell.cwd = str(tmp_path)
    provider = NodeProvider()
    result = await provider.collect(shell)
    assert result is not None
    assert result.payload.get("scripts") == {}
    assert result.payload.get("dependencies") == {}
    assert result.payload.get("dev_dependencies") == {}


async def test_node_provider_parses_package_json(
    shell: MagicMock, tmp_path: Path
) -> None:
    """NodeProvider parses scripts, dependencies, and devDependencies."""
    (tmp_path / "package.json").write_text(
        """
        {
            "name": "example",
            "scripts": {"build": "tsc", "test": "vitest"},
            "dependencies": {"react": "^18.2.0"},
            "devDependencies": {"typescript": "^5.4.0"}
        }
        """
    )
    shell.execute = AsyncMock(
        return_value=CommandResult(
            stdout="v20.11.0\n",
            stderr="",
            exit_code=0,
            duration_ms=1,
            cwd=str(tmp_path),
        )
    )
    shell.cwd = str(tmp_path)
    provider = NodeProvider()
    result = await provider.collect(shell)
    assert result is not None
    assert result.payload.get("scripts") == {"build": "tsc", "test": "vitest"}
    assert result.payload.get("dependencies") == {"react": "^18.2.0"}
    assert result.payload.get("dev_dependencies") == {"typescript": "^5.4.0"}


async def test_node_provider_defaults_missing_package_json_keys(
    shell: MagicMock, tmp_path: Path
) -> None:
    """Missing scripts/dependencies/devDependencies keys degrade to {}."""
    (tmp_path / "package.json").write_text('{"name": "example"}')
    shell.execute = AsyncMock(
        return_value=CommandResult(
            stdout="v20.11.0\n",
            stderr="",
            exit_code=0,
            duration_ms=1,
            cwd=str(tmp_path),
        )
    )
    shell.cwd = str(tmp_path)
    provider = NodeProvider()
    result = await provider.collect(shell)
    assert result is not None
    assert result.payload.get("scripts") == {}
    assert result.payload.get("dependencies") == {}
    assert result.payload.get("dev_dependencies") == {}


async def test_node_provider_handles_malformed_package_json(
    shell: MagicMock, tmp_path: Path
) -> None:
    """NodeProvider does not crash on malformed package.json."""
    (tmp_path / "package.json").write_text("{not valid json")
    shell.execute = AsyncMock(
        return_value=CommandResult(
            stdout="v20.11.0\n",
            stderr="",
            exit_code=0,
            duration_ms=1,
            cwd=str(tmp_path),
        )
    )
    shell.cwd = str(tmp_path)
    provider = NodeProvider()
    result = await provider.collect(shell)
    assert result is not None
    assert result.payload.get("node_version") == "20.11.0"
    assert result.payload.get("scripts") == {}


async def test_node_provider_commands_are_shell_portable(
    shell: MagicMock, tmp_path: Path
) -> None:
    """NodeProvider never embeds POSIX-only redirection in its commands."""
    commands: list[str] = []

    async def execute(command: str) -> CommandResult:
        commands.append(command)
        return CommandResult(
            stdout="v20.11.0\n",
            stderr="",
            exit_code=0,
            duration_ms=1,
            cwd=str(tmp_path),
        )

    shell.execute = execute
    shell.cwd = str(tmp_path)
    await NodeProvider().collect(shell)
    assert commands
    assert all("/dev/null" not in c for c in commands)
