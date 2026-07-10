"""Contract tests for OpenaiAgent using a mocked HTTP client."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from agentsh.agent.openai import OpenaiAgent

from agentsh.config import AgentConfig
from agentsh.context.sanitize import render_context_fragment
from agentsh.models import ContextFragment, Message

FRAGMENT_SPY_TARGET = "agentsh.agent._system.render_context_fragment"


def _fragment(provider: str = "git") -> ContextFragment:
    """Build a minimal context fragment for cache-identity tests."""
    return ContextFragment(provider=provider, summary="on main", payload={})


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


async def test_respond_tolerates_malformed_tool_arguments(
    config: AgentConfig, tool_use_response: MagicMock
) -> None:
    """Truncated arguments JSON yields empty args instead of crashing."""
    tool_use_response.choices[0].message.tool_calls[
        0
    ].function.arguments = '{"command": "ls'
    agent = OpenaiAgent(config)
    with patch.object(
        agent._client.chat.completions,
        "create",
        new=AsyncMock(return_value=tool_use_response),
    ):
        result = await agent.respond(
            conversation=[Message(role="user", content="list files")],
            context=[],
            tools=[],
        )
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].arguments == {}


async def test_respond_sends_a_stable_prompt_cache_key(
    config: AgentConfig, text_response: MagicMock
) -> None:
    """A stable prompt_cache_key is sent on every call so OpenAI's
    automatic prompt caching routes repeat requests to the same
    cache-holding backend.
    """
    agent = OpenaiAgent(config)
    mock_create = AsyncMock(return_value=text_response)
    with patch.object(
        agent._client.chat.completions, "create", new=mock_create
    ):
        await agent.respond(conversation=[], context=[], tools=[])
        first_key = mock_create.call_args.kwargs["prompt_cache_key"]
        await agent.respond(conversation=[], context=[], tools=[])
        second_key = mock_create.call_args.kwargs["prompt_cache_key"]

    assert isinstance(first_key, str)
    assert first_key
    assert first_key == second_key


async def test_respond_sends_web_search_options_when_web_fetch_enabled(
    monkeypatch: pytest.MonkeyPatch, text_response: MagicMock
) -> None:
    """web_fetch=True enables OpenAI's web search / browsing tool."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    config = AgentConfig(model="gpt-4o", web_fetch=True)
    agent = OpenaiAgent(config)
    mock_create = AsyncMock(return_value=text_response)
    with patch.object(
        agent._client.chat.completions, "create", new=mock_create
    ):
        await agent.respond(conversation=[], context=[], tools=[])

    assert mock_create.call_args.kwargs["web_search_options"] == {}


async def test_respond_omits_web_search_options_by_default(
    config: AgentConfig, text_response: MagicMock
) -> None:
    """web_fetch=False (the default) does not enable web search."""
    agent = OpenaiAgent(config)
    mock_create = AsyncMock(return_value=text_response)
    with patch.object(
        agent._client.chat.completions, "create", new=mock_create
    ):
        await agent.respond(conversation=[], context=[], tools=[])

    sent = mock_create.call_args.kwargs["web_search_options"]
    assert bool(sent) is False


async def test_respond_reuses_system_message_for_same_context_object(
    config: AgentConfig, text_response: MagicMock
) -> None:
    """The system message is rebuilt once per turn (same context list
    object), not once per loop iteration.
    """
    agent = OpenaiAgent(config)
    mock_create = AsyncMock(return_value=text_response)
    context = [_fragment()]

    with (
        patch.object(agent._client.chat.completions, "create", new=mock_create),
        patch(FRAGMENT_SPY_TARGET, wraps=render_context_fragment) as spy,
    ):
        await agent.respond(conversation=[], context=context, tools=[])
        await agent.respond(conversation=[], context=context, tools=[])

    assert spy.call_count == 1


async def test_respond_reuses_tools_for_same_tools_object(
    config: AgentConfig, text_response: MagicMock
) -> None:
    """The tool list is converted once per turn (same tools list object),
    not once per loop iteration.
    """
    agent = OpenaiAgent(config)
    mock_create = AsyncMock(return_value=text_response)
    tools = [
        {
            "name": "RunCommand",
            "description": "run a command",
            "input_schema": {},  # type: ignore[typeddict-item]
        }
    ]

    with patch.object(
        agent._client.chat.completions, "create", new=mock_create
    ):
        await agent.respond(conversation=[], context=[], tools=tools)  # type: ignore[arg-type]
        first_tools = mock_create.call_args.kwargs["tools"]
        await agent.respond(conversation=[], context=[], tools=tools)  # type: ignore[arg-type]
        second_tools = mock_create.call_args.kwargs["tools"]

    assert first_tools is second_tools
