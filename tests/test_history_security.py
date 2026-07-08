"""Tests for history file hardening helpers in history_security."""

import stat
from pathlib import Path

import pytest

from agentsh.history_security import (
    HISTORY_FILE_MODE,
    append_secure_line,
    ensure_secure_file,
    env_flag_enabled,
)


def _mode(path: Path) -> int:
    """Return the permission bits of path as an octal-comparable int."""
    return stat.S_IMODE(path.stat().st_mode)


class TestAppendSecureLine:
    """Tests for append_secure_line."""

    def test_creates_file_with_secure_mode(self, tmp_path: Path) -> None:
        """A freshly created file is mode 0o600."""
        target = tmp_path / "hist"
        append_secure_line(target, "echo hi")
        assert _mode(target) == HISTORY_FILE_MODE

    def test_appends_content_with_trailing_newline(
        self, tmp_path: Path
    ) -> None:
        """The written line is present with a trailing newline."""
        target = tmp_path / "hist"
        append_secure_line(target, "echo hi")
        assert target.read_text() == "echo hi\n"

    def test_multiple_calls_accumulate_lines(self, tmp_path: Path) -> None:
        """Repeated calls append rather than overwrite."""
        target = tmp_path / "hist"
        append_secure_line(target, "first")
        append_secure_line(target, "second")
        append_secure_line(target, "third")
        assert target.read_text() == "first\nsecond\nthird\n"

    def test_creates_missing_parent_directories(self, tmp_path: Path) -> None:
        """Nonexistent parent directories are created as needed."""
        target = tmp_path / "a" / "b" / "c" / "hist"
        append_secure_line(target, "echo hi")
        assert target.exists()
        assert target.read_text() == "echo hi\n"

    def test_rehardens_preexisting_loose_permissions(
        self, tmp_path: Path
    ) -> None:
        """A file that already exists with a looser mode gets re-hardened."""
        target = tmp_path / "hist"
        target.write_text("old line\n")
        target.chmod(0o644)
        assert _mode(target) == 0o644

        append_secure_line(target, "new line")

        assert _mode(target) == HISTORY_FILE_MODE
        assert target.read_text() == "old line\nnew line\n"


class TestEnsureSecureFile:
    """Tests for ensure_secure_file."""

    def test_creates_empty_file_with_secure_mode(self, tmp_path: Path) -> None:
        """A fresh file is created empty and mode 0o600."""
        target = tmp_path / "hist"
        ensure_secure_file(target)
        assert target.exists()
        assert target.read_text() == ""
        assert _mode(target) == HISTORY_FILE_MODE

    def test_creates_missing_parent_directories(self, tmp_path: Path) -> None:
        """Nonexistent parent directories are created as needed."""
        target = tmp_path / "a" / "b" / "hist"
        ensure_secure_file(target)
        assert target.exists()

    def test_rehardens_preexisting_loose_permissions(
        self, tmp_path: Path
    ) -> None:
        """A pre-existing file with a looser mode gets re-hardened."""
        target = tmp_path / "hist"
        target.write_text("existing content\n")
        target.chmod(0o644)

        ensure_secure_file(target)

        assert _mode(target) == HISTORY_FILE_MODE
        assert target.read_text() == "existing content\n"

    def test_does_not_write_content(self, tmp_path: Path) -> None:
        """Calling on a fresh path leaves the file empty."""
        target = tmp_path / "hist"
        ensure_secure_file(target)
        assert target.stat().st_size == 0


class TestEnvFlagEnabled:
    """Tests for env_flag_enabled."""

    @pytest.mark.parametrize(
        "value", ["1", "true", "yes", "on", "TRUE", "YES", "On", "  1  "]
    )
    def test_truthy_values(
        self, monkeypatch: pytest.MonkeyPatch, value: str
    ) -> None:
        """Truthy spellings, case-insensitive and stripped, return True."""
        monkeypatch.setenv("AGENTSH_TEST_FLAG", value)
        assert env_flag_enabled("AGENTSH_TEST_FLAG") is True

    @pytest.mark.parametrize("value", ["", "0", "false", "no", "off", "xyz"])
    def test_falsy_values(
        self, monkeypatch: pytest.MonkeyPatch, value: str
    ) -> None:
        """Falsy or unrecognized values return False."""
        monkeypatch.setenv("AGENTSH_TEST_FLAG", value)
        assert env_flag_enabled("AGENTSH_TEST_FLAG") is False

    def test_unset_returns_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An unset environment variable returns False."""
        monkeypatch.delenv("AGENTSH_TEST_FLAG", raising=False)
        assert env_flag_enabled("AGENTSH_TEST_FLAG") is False
