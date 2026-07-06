"""Tests for ReadFile tool."""

from pathlib import Path

import pytest
from agentsh.tools.read_file import ReadFile


async def test_read_existing_file(tmp_path: Path) -> None:
    """invoke returns the file contents."""
    f = tmp_path / "hello.txt"
    f.write_text("hello world")
    tool = ReadFile()
    result = await tool.invoke(path=str(f))
    assert result == "hello world"


async def test_read_missing_file_raises(tmp_path: Path) -> None:
    """invoke raises FileNotFoundError for a missing file."""
    tool = ReadFile()
    with pytest.raises(FileNotFoundError):
        await tool.invoke(path=str(tmp_path / "missing.txt"))


async def test_read_file_expands_home(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Tilde paths resolve to the user's home directory."""
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / "notes.txt").write_text("home sweet home", encoding="utf-8")
    result = await ReadFile().invoke(path="~/notes.txt")
    assert result == "home sweet home"


async def test_read_non_ascii_content(tmp_path: Path) -> None:
    """Non-ASCII files are decoded as UTF-8 regardless of locale."""
    f = tmp_path / "unicode.txt"
    f.write_text("café ☕", encoding="utf-8")
    assert await ReadFile().invoke(path=str(f)) == "café ☕"
