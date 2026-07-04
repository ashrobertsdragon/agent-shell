"""Tests for GitProvider and FilesystemProvider."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from agentsh.context.providers.filesystem import FilesystemProvider
from agentsh.context.providers.git import GitProvider
from agentsh.models import CommandResult


@pytest.fixture
def shell() -> MagicMock:
    """Minimal shell mock."""
    return MagicMock()


async def test_git_provider_returns_fragment_in_git_repo(shell: MagicMock) -> None:
    """GitProvider returns a fragment when inside a git repo."""
    shell.execute = AsyncMock(
        side_effect=[
            CommandResult(
                stdout="main\n", stderr="", exit_code=0, duration_ms=1, cwd="/repo"
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
    shell.cwd = AsyncMock(return_value=str(tmp_path))
    (tmp_path / "main.py").touch()
    provider = FilesystemProvider()
    result = await provider.collect(shell)
    assert result is not None
    assert "main.py" in result.payload.get("files", [])
