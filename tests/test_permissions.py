"""Tests for the permission engine, including deny-precedence."""

import pytest

from agentsh.config import PermissionRulesConfig
from agentsh.permissions import PermissionEngine, PermissionLevel


@pytest.fixture
def engine() -> PermissionEngine:
    """Permission engine with representative allow/confirm/deny rules."""
    rules = PermissionRulesConfig(
        allow={"RunCommand:ls*", "RunCommand:pwd", "ReadFile:*"},
        confirm={"RunCommand:git commit*", "WriteFile:*"},
        deny={"RunCommand:rm -rf*"},
    )
    return PermissionEngine(rules)


def test_allow(engine: PermissionEngine) -> None:
    assert engine.evaluate("RunCommand:ls -la") == PermissionLevel.ALLOW


def test_confirm(engine: PermissionEngine) -> None:
    assert (
        engine.evaluate("RunCommand:git commit -m 'x'")
        == PermissionLevel.CONFIRM
    )


def test_deny(engine: PermissionEngine) -> None:
    assert engine.evaluate("RunCommand:rm -rf /") == PermissionLevel.DENY


def test_deny_beats_allow() -> None:
    """A deny pattern wins even when an allow pattern also matches."""
    rules = PermissionRulesConfig(
        allow={"RunCommand:rm*"},
        deny={"RunCommand:rm -rf*"},
    )
    engine = PermissionEngine(rules)
    assert engine.evaluate("RunCommand:rm -rf /tmp/x") == PermissionLevel.DENY


def test_default_is_confirm(engine: PermissionEngine) -> None:
    """Unmatched keys default to CONFIRM, not ALLOW."""
    assert engine.evaluate("RunCommand:unknown-tool") == PermissionLevel.CONFIRM


def test_read_file_allow(engine: PermissionEngine) -> None:
    assert engine.evaluate("ReadFile:file") == PermissionLevel.ALLOW


def test_write_file_confirm(engine: PermissionEngine) -> None:
    assert engine.evaluate("WriteFile") == PermissionLevel.CONFIRM
