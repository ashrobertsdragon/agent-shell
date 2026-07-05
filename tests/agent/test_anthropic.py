"""Contract tests for AnthropicAgent using a mocked HTTP client."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from agentsh.agent.anthropic import AnthropicAgent

from agentsh.config import AgentConfig
from agentsh.models import Message


@pytest.fixture
def config() -> AgentConfig:
    """Minimal agent backend config for testing."""
    return AgentConfig(model="claude-haiku-4-5-20251001", web_fetch=False)


@pytest.fixture
def text_response() -> MagicMock:
    """Simulated Anthropic text-only response."""
    block = MagicMock()
    block.type = "text"
    block.text = "Hello from the agent."
    response = MagicMock()
    response.content = [block]
    return response


@pytest.fixture
def tool_use_response() -> MagicMock:
    """Simulated Anthropic tool_use response."""
    block = MagicMock()
    block.type = "tool_use"
    block.id = "tu_abc"
    block.name = "RunCommand"
    block.input = {"command": "ls -la"}
    response = MagicMock()
    response.content = [block]
    return response


async def test_respond_returns_text_message(
    config: AgentConfig, text_response: MagicMock
) -> None:
    """respond converts a text block into an assistant Message."""
    agent = AnthropicAgent(config)
    with patch.object(
        agent._client.messages,
        "create",
        new=AsyncMock(return_value=text_response),
    ):
        result = await agent.respond(
            conversation=[Message(role="user", content="hello")],
            context=[],
            tools=[],
        )
    assert result.role == "assistant"
    assert result.content == "Hello from the agent."
    assert result.tool_calls == ()


async def test_respond_parses_tool_calls(
    config: AgentConfig, tool_use_response: MagicMock
) -> None:
    """respond converts tool_use blocks into ToolCall objects."""
    agent = AnthropicAgent(config)
    with patch.object(
        agent._client.messages,
        "create",
        new=AsyncMock(return_value=tool_use_response),
    ):
        result = await agent.respond(
            conversation=[Message(role="user", content="list files")],
            context=[],
            tools=[
                {
                    "name": "RunCommand",
                    "description": "run a command",
                    "input_schema": {},  # type: ignore[typeddict-item]
                }
            ],
        )
    assert len(result.tool_calls) == 1
    tc = result.tool_calls[0]
    assert tc.tool_name == "RunCommand"
    assert tc.arguments == {"command": "ls -la"}
    assert tc.call_id == "tu_abc"
