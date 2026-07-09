"""Tests for shared output-size limits and bounded file reads."""

from pathlib import Path
from typing import IO

import pytest

from agentsh.limits import read_last_lines


class _CountingReader:
    """Wraps a binary file object, tallying bytes returned from read().

    Real file objects (io.BufferedReader) are immutable C types whose
    methods can't be monkeypatched directly, so this wraps one instead.
    """

    def __init__(self, fh: IO[bytes]) -> None:
        """Store the underlying file object and reset the byte tally."""
        self._fh = fh
        self.bytes_read = 0

    def __enter__(self) -> "_CountingReader":
        """Return self so the wrapper is usable as a context manager."""
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Close the underlying file object."""
        self._fh.close()

    def seek(self, offset: int, whence: int = 0) -> int:
        """Delegate to the underlying file object's seek."""
        return self._fh.seek(offset, whence)

    def tell(self) -> int:
        """Delegate to the underlying file object's tell."""
        return self._fh.tell()

    def read(self, size: int = -1) -> bytes:
        """Delegate to the underlying file object's read, tallying bytes."""
        data = self._fh.read(size)
        self.bytes_read += len(data)
        return data


def test_read_last_lines_returns_exact_tail(tmp_path: Path) -> None:
    """The last N lines are returned in original order, oldest first."""
    path = tmp_path / "history"
    lines = [f"cmd-{i}" for i in range(50)]
    path.write_text("\n".join(lines) + "\n")

    assert read_last_lines(path, 5) == lines[-5:]


def test_read_last_lines_limit_exceeds_file_length(tmp_path: Path) -> None:
    """Requesting more lines than exist returns the whole file."""
    path = tmp_path / "history"
    lines = ["a", "b", "c"]
    path.write_text("\n".join(lines) + "\n")

    assert read_last_lines(path, 100) == lines


def test_read_last_lines_missing_file_raises(tmp_path: Path) -> None:
    """A missing file raises FileNotFoundError, same as Path.read_text."""
    with pytest.raises(FileNotFoundError):
        read_last_lines(tmp_path / "missing", 10)


def test_read_last_lines_zero_limit_returns_empty(tmp_path: Path) -> None:
    """A limit of zero returns an empty list without opening the file."""
    assert read_last_lines(tmp_path / "missing", 0) == []


def test_read_last_lines_does_not_read_entire_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A small tail request on a huge file reads only a small fraction of it.

    Regression test for #15: history() previously did
    ``Path(histfile).read_text().splitlines()[-limit:]``, so per-turn cost
    scaled with the file's total lifetime size instead of the requested
    tail size. This asserts the fix reads a bounded amount of data
    regardless of how large the file is.
    """
    path = tmp_path / "big_history"
    line = "x" * 40
    num_lines = 200_000
    with path.open("w") as f:
        for _ in range(num_lines):
            f.write(line + "\n")
    file_size = path.stat().st_size
    assert file_size > 5_000_000

    readers: list[_CountingReader] = []
    real_open = Path.open

    def counting_open(self: Path, mode: str = "r") -> _CountingReader:
        reader = _CountingReader(real_open(self, mode))
        readers.append(reader)
        return reader

    monkeypatch.setattr(Path, "open", counting_open)

    result = read_last_lines(path, 100)
    bytes_read = sum(reader.bytes_read for reader in readers)

    assert result == [line] * 100
    assert bytes_read < file_size // 10
