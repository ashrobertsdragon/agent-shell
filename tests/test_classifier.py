"""Table-driven tests for the input classifier."""

from unittest.mock import MagicMock

import pytest
from agentsh.classifier import InputKind, classify


@pytest.fixture
def shell() -> MagicMock:
    """Shell mock that parses common shell commands."""
    s = MagicMock()
    s.can_parse = MagicMock(
        side_effect=lambda raw: raw.startswith(("ls", "cd", "echo", "git", "pwd"))
    )
    return s


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("/agent tell me a joke", InputKind.AGENT),
        ("ls -la", InputKind.SHELL_PARSEABLE),
        ("echo hello world", InputKind.SHELL_PARSEABLE),
        ("show me all python files", InputKind.AGENT),
        ("what does this repo do", InputKind.AGENT),
        ("/agent ", InputKind.AGENT),
    ],
)
def test_classify(raw: str, expected: InputKind, shell: MagicMock) -> None:
    """classify routes correctly based on prefix and shell parse result."""
    assert classify(raw, shell) == expected
