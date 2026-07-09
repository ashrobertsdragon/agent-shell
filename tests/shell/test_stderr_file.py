"""Tests for the shared stderr scratch-file helpers."""

from pathlib import Path

from agentsh.shell.plugin._stderr_file import (
    create_stderr_tempfile,
    discard_stderr_tempfile,
)


def test_create_stderr_tempfile_returns_existing_empty_file() -> None:
    """create_stderr_tempfile returns a path to a fresh, empty file."""
    path = create_stderr_tempfile()
    try:
        p = Path(path)
        assert p.exists()
        assert p.read_bytes() == b""
        assert p.name.startswith("agentsh_stderr_")
    finally:
        discard_stderr_tempfile(path)


def test_discard_stderr_tempfile_removes_file() -> None:
    """discard_stderr_tempfile removes the file it is given."""
    path = create_stderr_tempfile()
    discard_stderr_tempfile(path)
    assert not Path(path).exists()


def test_discard_stderr_tempfile_is_idempotent(tmp_path: Path) -> None:
    """Discarding an already-removed path does not raise."""
    missing = tmp_path / "already-gone"
    discard_stderr_tempfile(str(missing))
