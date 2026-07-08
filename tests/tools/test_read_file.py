"""Tests for ReadFile tool."""

from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from agentsh.config import PermissionRulesConfig
from agentsh.permissions import PermissionDeniedError, PermissionEngine
from agentsh.tools.read_file import ReadFile


@pytest.fixture
def allow_all() -> PermissionEngine:
    """PermissionEngine that ALLOWs every ReadFile call."""
    return PermissionEngine(PermissionRulesConfig(allow={"ReadFile:*"}))


async def test_read_existing_file(
    tmp_path: Path, allow_all: PermissionEngine
) -> None:
    """invoke returns the file contents."""
    f = tmp_path / "hello.txt"
    f.write_text("hello world")
    tool = ReadFile(permissions=allow_all)
    result = await tool.invoke(path=str(f))
    assert result == "hello world"


async def test_read_missing_file_raises(
    tmp_path: Path, allow_all: PermissionEngine
) -> None:
    """invoke raises FileNotFoundError for a missing file."""
    tool = ReadFile(permissions=allow_all)
    with pytest.raises(FileNotFoundError):
        await tool.invoke(path=str(tmp_path / "missing.txt"))


async def test_read_file_expands_home(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    allow_all: PermissionEngine,
) -> None:
    """Tilde paths resolve to the user's home directory."""
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / "notes.txt").write_text("home sweet home", encoding="utf-8")
    result = await ReadFile(permissions=allow_all).invoke(path="~/notes.txt")
    assert result == "home sweet home"


async def test_read_non_ascii_content(
    tmp_path: Path, allow_all: PermissionEngine
) -> None:
    """Non-ASCII files are decoded as UTF-8 regardless of locale."""
    f = tmp_path / "unicode.txt"
    f.write_text("café ☕", encoding="utf-8")
    assert await ReadFile(permissions=allow_all).invoke(path=str(f)) == (
        "café ☕"
    )


async def test_read_deny_raises_without_touching_file(
    tmp_path: Path,
) -> None:
    """A DENY-matched path raises PermissionDeniedError and never reads."""
    f = tmp_path / "secret.txt"
    f.write_text("top secret")
    key = f"ReadFile:{f.resolve().as_posix()}"
    permissions = PermissionEngine(PermissionRulesConfig(deny={key}))
    tool = ReadFile(permissions=permissions)
    with pytest.raises(PermissionDeniedError):
        await tool.invoke(path=str(f))


async def test_read_confirm_blocks_without_callback(tmp_path: Path) -> None:
    """A CONFIRM-matched path is blocked when the tool is invoked directly
    with no confirm callback wired in, bypassing the agent loop entirely.
    """
    f = tmp_path / "secret.txt"
    f.write_text("top secret")
    key = f"ReadFile:{f.resolve().as_posix()}"
    permissions = PermissionEngine(PermissionRulesConfig(confirm={key}))
    tool = ReadFile(permissions=permissions)
    with pytest.raises(PermissionDeniedError):
        await tool.invoke(path=str(f))


async def test_read_confirm_proceeds_when_callback_approves(
    tmp_path: Path,
) -> None:
    """A CONFIRM-matched path reads once the confirm callback approves it."""
    f = tmp_path / "secret.txt"
    f.write_text("top secret")
    key = f"ReadFile:{f.resolve().as_posix()}"
    permissions = PermissionEngine(PermissionRulesConfig(confirm={key}))
    confirm = AsyncMock(return_value=True)
    tool = ReadFile(permissions=permissions, confirm=confirm)
    assert await tool.invoke(path=str(f)) == "top secret"


async def test_read_confirm_blocks_when_callback_declines(
    tmp_path: Path,
) -> None:
    """A CONFIRM-matched path is blocked when the confirm callback declines."""
    f = tmp_path / "secret.txt"
    f.write_text("top secret")
    key = f"ReadFile:{f.resolve().as_posix()}"
    permissions = PermissionEngine(PermissionRulesConfig(confirm={key}))
    confirm = AsyncMock(return_value=False)
    tool = ReadFile(permissions=permissions, confirm=confirm)
    with pytest.raises(PermissionDeniedError):
        await tool.invoke(path=str(f))
    confirm.assert_awaited_once()
