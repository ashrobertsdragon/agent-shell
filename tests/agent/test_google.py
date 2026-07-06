"""Contract tests for GoogleAgent using a mocked HTTP client."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from agentsh.agent.google import GoogleAgent

from agentsh.config import AgentConfig
from agentsh.models import Message


@pytest.fixture
def config() -> AgentConfig:
    """Minimal agent backend config for testing."""
    return AgentConfig(model="gemini-2.0-flash", web_fetch=False)


@pytest.fixture
def text_response() -> MagicMock:
    """Simulated Google GenAI text-only response."""
    part = MagicMock()
    part.text = "Hello from the agent."
    part.function_call = None

    content = MagicMock()
    content.parts = [part]

    candidate = MagicMock()
    candidate.content = content

    response = MagicMock()
    response.candidates = [candidate]
    return response


@pytest.fixture
def tool_use_response() -> MagicMock:
    """Simulated Google GenAI function_call response."""
    function_call = MagicMock()
    function_call.name = "RunCommand"
    function_call.args = {"command": "ls -la"}
    function_call.id = "tu_abc"

    part = MagicMock()
    part.text = None
    part.function_call = function_call

    content = MagicMock()
    content.parts = [part]

    candidate = MagicMock()
    candidate.content = content

    response = MagicMock()
    response.candidates = [candidate]
    return response


async def test_respond_returns_text_message(
    config: AgentConfig, text_response: MagicMock
) -> None:
    """respond converts a text response into an assistant Message."""
    agent = GoogleAgent(config)
    with patch.object(
        agent._client.aio.models,
        "generate_content",
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
    """respond converts function_call blocks into ToolCall objects."""
    agent = GoogleAgent(config)
    with patch.object(
        agent._client.aio.models,
        "generate_content",
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
