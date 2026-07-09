"""Tests for WriteFile tool."""

from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from agentsh.config import PermissionRulesConfig
from agentsh.permissions import PermissionDeniedError, PermissionEngine
from agentsh.tools.write_file import WriteFile


@pytest.fixture
def allow_all() -> PermissionEngine:
    """PermissionEngine that ALLOWs every WriteFile call."""
    return PermissionEngine(PermissionRulesConfig(allow={"WriteFile:*"}))


async def test_full_write(tmp_path: Path, allow_all: PermissionEngine) -> None:
    """invoke with content overwrites the file."""
    f = tmp_path / "out.txt"
    tool = WriteFile(permissions=allow_all)
    await tool.invoke(path=str(f), content="new content")
    assert f.read_text() == "new content"


async def test_patch_replaces_block(
    tmp_path: Path, allow_all: PermissionEngine
) -> None:
    """invoke with patch applies SEARCH/REPLACE to the file."""
    f = tmp_path / "code.py"
    f.write_text("def foo():\n    return 1\n")
    tool = WriteFile(permissions=allow_all)
    patch = (
        "<<<<<<< SEARCH\n    return 1\n=======\n    return 42\n>>>>>>> REPLACE"
    )
    await tool.invoke(path=str(f), patch=patch)
    assert "return 42" in f.read_text()


async def test_patch_raises_if_search_not_found(
    tmp_path: Path, allow_all: PermissionEngine
) -> None:
    """invoke raises ValueError when SEARCH text is absent."""
    f = tmp_path / "code.py"
    f.write_text("def foo(): pass\n")
    tool = WriteFile(permissions=allow_all)
    patch = "<<<<<<< SEARCH\nmissing\n=======\nreplaced\n>>>>>>> REPLACE"
    with pytest.raises(ValueError, match="not found"):
        await tool.invoke(path=str(f), patch=patch)


async def test_requires_content_or_patch(
    tmp_path: Path, allow_all: PermissionEngine
) -> None:
    """invoke raises ValueError if neither content nor patch is supplied."""
    tool = WriteFile(permissions=allow_all)
    with pytest.raises(ValueError, match="content or patch"):
        await tool.invoke(path=str(tmp_path / "x.txt"))


async def test_write_file_expands_home(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    allow_all: PermissionEngine,
) -> None:
    """Tilde paths resolve to the user's home directory."""
    monkeypatch.setenv("HOME", str(tmp_path))
    await WriteFile(permissions=allow_all).invoke(
        path="~/out.txt", content="written"
    )
    assert (tmp_path / "out.txt").read_text(encoding="utf-8") == "written"


async def test_write_and_patch_preserve_non_ascii_content(
    tmp_path: Path, allow_all: PermissionEngine
) -> None:
    """Non-ASCII content round-trips as UTF-8 regardless of locale."""
    f = tmp_path / "unicode.txt"
    await WriteFile(permissions=allow_all).invoke(
        path=str(f), content="café ☕ emoji 🎉"
    )
    assert f.read_text(encoding="utf-8") == "café ☕ emoji 🎉"
    patch = (
        "<<<<<<< SEARCH\ncafé ☕ emoji 🎉\n=======\nnaïve 🌊\n>>>>>>> REPLACE"
    )
    await WriteFile(permissions=allow_all).invoke(path=str(f), patch=patch)
    assert f.read_text(encoding="utf-8") == "naïve 🌊"


async def test_write_deny_raises_without_touching_file(
    tmp_path: Path,
) -> None:
    """A DENY-matched path raises PermissionDeniedError and never writes."""
    f = tmp_path / "protected.txt"
    key = f"WriteFile:{f.resolve().as_posix()}"
    permissions = PermissionEngine(PermissionRulesConfig(deny={key}))
    tool = WriteFile(permissions=permissions)
    with pytest.raises(PermissionDeniedError):
        await tool.invoke(path=str(f), content="malicious")
    assert not f.exists()


async def test_write_confirm_blocks_without_callback(tmp_path: Path) -> None:
    """A CONFIRM-matched path is blocked when the tool is invoked directly
    with no confirm callback wired in, bypassing the agent loop entirely.
    """
    f = tmp_path / "protected.txt"
    key = f"WriteFile:{f.resolve().as_posix()}"
    permissions = PermissionEngine(PermissionRulesConfig(confirm={key}))
    tool = WriteFile(permissions=permissions)
    with pytest.raises(PermissionDeniedError):
        await tool.invoke(path=str(f), content="unattended write")
    assert not f.exists()


async def test_write_confirm_proceeds_when_callback_approves(
    tmp_path: Path,
) -> None:
    """A CONFIRM-matched path writes once the confirm callback approves it."""
    f = tmp_path / "protected.txt"
    key = f"WriteFile:{f.resolve().as_posix()}"
    permissions = PermissionEngine(PermissionRulesConfig(confirm={key}))
    confirm = AsyncMock(return_value=True)
    tool = WriteFile(permissions=permissions, confirm=confirm)
    await tool.invoke(path=str(f), content="approved write")
    assert f.read_text() == "approved write"


async def test_write_confirm_blocks_when_callback_declines(
    tmp_path: Path,
) -> None:
    """A CONFIRM-matched path is blocked when the confirm callback declines."""
    f = tmp_path / "protected.txt"
    key = f"WriteFile:{f.resolve().as_posix()}"
    permissions = PermissionEngine(PermissionRulesConfig(confirm={key}))
    confirm = AsyncMock(return_value=False)
    tool = WriteFile(permissions=permissions, confirm=confirm)
    with pytest.raises(PermissionDeniedError):
        await tool.invoke(path=str(f), content="unattended write")
    assert not f.exists()
    confirm.assert_awaited_once()


async def test_write_outside_sandbox_root_denied_even_when_allowed(
    tmp_path: Path,
) -> None:
    """A path outside the configured sandbox roots is refused even though
    the permission engine ALLOWs every WriteFile call.

    This is the defense-in-depth gap from issue #21: once a call clears
    the (glob-based) permission rules, nothing stopped the tool from
    creating arbitrary directory trees anywhere the process can write.
    """
    sandbox = tmp_path / "workspace"
    sandbox.mkdir()
    outside = tmp_path / "outside" / "escaped.txt"
    permissions = PermissionEngine(PermissionRulesConfig(allow={"WriteFile:*"}))
    tool = WriteFile(permissions=permissions, sandbox_roots=[sandbox])

    with pytest.raises(PermissionDeniedError):
        await tool.invoke(path=str(outside), content="escaped")

    assert not outside.exists()
    assert not outside.parent.exists()


async def test_write_inside_sandbox_root_succeeds(tmp_path: Path) -> None:
    """A path inside a configured sandbox root writes normally."""
    sandbox = tmp_path / "workspace"
    sandbox.mkdir()
    target = sandbox / "nested" / "file.txt"
    permissions = PermissionEngine(PermissionRulesConfig(allow={"WriteFile:*"}))
    tool = WriteFile(permissions=permissions, sandbox_roots=[sandbox])

    await tool.invoke(path=str(target), content="hello")

    assert target.read_text() == "hello"


async def test_write_sandbox_root_itself_is_writable(tmp_path: Path) -> None:
    """A file written directly at the sandbox root (not nested) is allowed."""
    sandbox = tmp_path / "workspace"
    sandbox.mkdir()
    target = sandbox / "top-level.txt"
    permissions = PermissionEngine(PermissionRulesConfig(allow={"WriteFile:*"}))
    tool = WriteFile(permissions=permissions, sandbox_roots=[sandbox])

    await tool.invoke(path=str(target), content="hello")

    assert target.read_text() == "hello"


async def test_no_sandbox_roots_configured_is_unrestricted(
    tmp_path: Path, allow_all: PermissionEngine
) -> None:
    """With no sandbox roots configured, behavior is unchanged (opt-in)."""
    f = tmp_path / "anywhere" / "out.txt"
    tool = WriteFile(permissions=allow_all, sandbox_roots=None)
    await tool.invoke(path=str(f), content="new content")
    assert f.read_text() == "new content"
