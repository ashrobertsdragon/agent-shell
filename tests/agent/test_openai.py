"""Contract tests for OpenaiAgent using a mocked HTTP client."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from agentsh.agent.openai import OpenaiAgent

from agentsh.config import AgentConfig
from agentsh.models import Message


@pytest.fixture
def config(monkeypatch: pytest.MonkeyPatch) -> AgentConfig:
    """Minimal agent backend config for testing."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    return AgentConfig(model="gpt-4o", web_fetch=False)


@pytest.fixture
def text_response() -> MagicMock:
    """Simulated OpenAI text-only response."""
    message = MagicMock()
    message.content = "Hello from the agent."
    message.tool_calls = None
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    return response


@pytest.fixture
def tool_use_response() -> MagicMock:
    """Simulated OpenAI tool_calls response."""
    function_call = MagicMock()
    function_call.name = "RunCommand"
    function_call.arguments = '{"command": "ls -la"}'

    tc = MagicMock()
    tc.id = "tu_abc"
    tc.type = "function"
    tc.function = function_call

    message = MagicMock()
    message.content = ""
    message.tool_calls = [tc]

    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    return response


async def test_respond_returns_text_message(
    config: AgentConfig, text_response: MagicMock
) -> None:
    """respond converts a text response into an assistant Message."""
    agent = OpenaiAgent(config)
    with patch.object(
        agent._client.chat.completions,
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
    """respond converts tool_calls blocks into ToolCall objects."""
    agent = OpenaiAgent(config)
    with patch.object(
        agent._client.chat.completions,
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
