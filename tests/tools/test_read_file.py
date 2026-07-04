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
