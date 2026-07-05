"""Table-driven tests for the input classifier."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from agentsh.classifier import InputKind, classify


@pytest.fixture
def shell() -> MagicMock:
    """Shell mock that parses common shell commands."""
    s = MagicMock()
    s.can_parse = AsyncMock(
        side_effect=lambda raw: raw.startswith(
            ("ls", "cd", "echo", "git", "pwd")
        )
    )
    return s


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("/agent tell me a joke", InputKind.AGENT),
        ("ls -la", InputKind.SHELL),
        ("echo hello world", InputKind.SHELL),
        ("show me all python files", InputKind.AGENT),
        ("what does this repo do", InputKind.AGENT),
        ("/agent ", InputKind.AGENT),
    ],
)
async def test_classify(
    raw: str, expected: InputKind, shell: MagicMock
) -> None:
    """classify routes correctly based on prefix and shell parse result."""
    assert await classify(raw, shell) == expected
