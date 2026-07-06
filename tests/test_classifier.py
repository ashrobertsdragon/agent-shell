"""Table-driven tests for the input classifier."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from agentsh.classifier import InputKind, agent_query, classify


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
        ("/agent", InputKind.AGENT),
    ],
)
async def test_classify(
    raw: str, expected: InputKind, shell: MagicMock
) -> None:
    """classify routes correctly based on prefix and shell parse result."""
    assert await classify(raw, shell) == expected


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("/agent tell me a joke", "tell me a joke"),
        ("/agent", ""),
        ("/agent ", ""),
        ("what does this repo do", "what does this repo do"),
        ("/agentfoo bar", "/agentfoo bar"),
    ],
)
def test_agent_query_strips_prefix(raw: str, expected: str) -> None:
    """agent_query removes the /agent prefix and surrounding whitespace."""
    assert agent_query(raw) == expected
