"""Tests for the create_shell factory and unsupported-shell handling."""

import pytest

from agentsh.shell import UnsupportedShellError, create_shell


def test_create_shell_unsupported_name_raises_with_guidance() -> None:
    """An unregistered shell name produces a clear, actionable error."""
    with pytest.raises(UnsupportedShellError) as exc_info:
        create_shell("tcsh")
    message = str(exc_info.value)
    assert "tcsh" in message
    assert "bash" in message
    assert "config.toml" in message


def test_create_shell_auto_unsupported_posix_shell_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Auto-detection of an unsupported $SHELL fails gracefully."""
    monkeypatch.setenv("SHELL", "/usr/bin/tcsh")
    with pytest.raises(UnsupportedShellError, match="tcsh"):
        create_shell("auto")


def test_create_shell_auto_detects_fish(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Auto-detection of $SHELL=/usr/bin/fish resolves to FishShell."""
    monkeypatch.setenv("SHELL", "/usr/bin/fish")
    shell = create_shell("auto")
    assert type(shell).__name__ == "FishShell"


def test_create_shell_auto_detects_nu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Auto-detection of $SHELL=/usr/bin/nu resolves to NuShellShell."""
    monkeypatch.setenv("SHELL", "/usr/bin/nu")
    shell = create_shell("auto")
    assert type(shell).__name__ == "NuShellShell"


def test_create_shell_auto_undetectable_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Detection failure surfaces as UnsupportedShellError, not a traceback."""
    for var in ("SHELL", "PSModulePath", "CMDCMDLINE"):
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(UnsupportedShellError, match="config.toml"):
        create_shell("auto")


def test_create_shell_registered_name_returns_instance() -> None:
    """A registered shell name constructs its backend."""
    shell = create_shell("bash")
    assert type(shell).__name__ == "BashShell"
