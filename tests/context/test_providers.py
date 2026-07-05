"""Tests for context providers."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from agentsh.context.providers import (
    DockerProvider,
    EnvironmentProvider,
    FilesystemProvider,
    GitProvider,
    HistoryProvider,
    PythonProvider,
)
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
