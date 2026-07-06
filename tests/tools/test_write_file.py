"""Tests for WriteFile tool."""

from pathlib import Path

import pytest
from agentsh.tools.write_file import WriteFile


async def test_full_write(tmp_path: Path) -> None:
    """invoke with content overwrites the file."""
    f = tmp_path / "out.txt"
    tool = WriteFile()
    await tool.invoke(path=str(f), content="new content")
    assert f.read_text() == "new content"


async def test_patch_replaces_block(tmp_path: Path) -> None:
    """invoke with patch applies SEARCH/REPLACE to the file."""
    f = tmp_path / "code.py"
    f.write_text("def foo():\n    return 1\n")
    tool = WriteFile()
    patch = (
        "<<<<<<< SEARCH\n    return 1\n=======\n    return 42\n>>>>>>> REPLACE"
    )
    await tool.invoke(path=str(f), patch=patch)
    assert "return 42" in f.read_text()


async def test_patch_raises_if_search_not_found(tmp_path: Path) -> None:
    """invoke raises ValueError when SEARCH text is absent."""
    f = tmp_path / "code.py"
    f.write_text("def foo(): pass\n")
    tool = WriteFile()
    patch = "<<<<<<< SEARCH\nmissing\n=======\nreplaced\n>>>>>>> REPLACE"
    with pytest.raises(ValueError, match="not found"):
        await tool.invoke(path=str(f), patch=patch)


async def test_requires_content_or_patch(tmp_path: Path) -> None:
    """invoke raises ValueError if neither content nor patch is supplied."""
    tool = WriteFile()
    with pytest.raises(ValueError, match="content or patch"):
        await tool.invoke(path=str(tmp_path / "x.txt"))


async def test_write_file_expands_home(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Tilde paths resolve to the user's home directory."""
    monkeypatch.setenv("HOME", str(tmp_path))
    await WriteFile().invoke(path="~/out.txt", content="written")
    assert (tmp_path / "out.txt").read_text(encoding="utf-8") == "written"


async def test_write_and_patch_preserve_non_ascii_content(
    tmp_path: Path,
) -> None:
    """Non-ASCII content round-trips as UTF-8 regardless of locale."""
    f = tmp_path / "unicode.txt"
    await WriteFile().invoke(path=str(f), content="café ☕ emoji 🎉")
    assert f.read_text(encoding="utf-8") == "café ☕ emoji 🎉"
    patch = (
        "<<<<<<< SEARCH\ncafé ☕ emoji 🎉\n=======\nnaïve 🌊\n>>>>>>> REPLACE"
    )
    await WriteFile().invoke(path=str(f), patch=patch)
    assert f.read_text(encoding="utf-8") == "naïve 🌊"
