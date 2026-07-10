"""Contract tests for OpenrouterAgent using a mocked HTTP client."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from agentsh.agent.openrouter import OpenrouterAgent

from agentsh.config import AgentConfig
from agentsh.context.sanitize import render_context_fragment
from agentsh.models import ContextFragment, Message

FRAGMENT_SPY_TARGET = "agentsh.agent._system.render_context_fragment"


def _fragment(provider: str = "git") -> ContextFragment:
    """Build a minimal context fragment for cache-identity tests."""
    return ContextFragment(provider=provider, summary="on main", payload={})


@pytest.fixture
def config() -> AgentConfig:
    """Minimal agent backend config for testing."""
    return AgentConfig(model="openai/gpt-4o", web_fetch=False)


@pytest.fixture
def text_response() -> MagicMock:
    """Simulated OpenRouter text-only response."""
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
    """Simulated OpenRouter tool_calls response."""
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
    agent = OpenrouterAgent(config)
    with patch.object(
        agent._client.chat,
        "send_async",
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
    agent = OpenrouterAgent(config)
    with patch.object(
        agent._client.chat,
        "send_async",
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
    agent = OpenrouterAgent(config)
    with patch.object(
        agent._client.chat,
        "send_async",
        new=AsyncMock(return_value=tool_use_response),
    ):
        result = await agent.respond(
            conversation=[Message(role="user", content="list files")],
            context=[],
            tools=[],
        )
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].arguments == {}


async def test_respond_marks_system_prompt_as_cache_breakpoint(
    config: AgentConfig, text_response: MagicMock
) -> None:
    """The system message is sent as a content-part list with a cache
    breakpoint on its last part.
    """
    agent = OpenrouterAgent(config)
    mock_send = AsyncMock(return_value=text_response)
    with patch.object(agent._client.chat, "send_async", new=mock_send):
        await agent.respond(
            conversation=[Message(role="user", content="hello")],
            context=[],
            tools=[],
        )

    sys_content = mock_send.call_args.kwargs["messages"][0]["content"]
    assert isinstance(sys_content, list)
    assert sys_content[-1]["cache_control"] == {"type": "ephemeral"}


async def test_respond_marks_last_tool_as_cache_breakpoint(
    config: AgentConfig, text_response: MagicMock
) -> None:
    """Only the last tool definition carries a cache breakpoint."""
    agent = OpenrouterAgent(config)
    mock_send = AsyncMock(return_value=text_response)
    with patch.object(agent._client.chat, "send_async", new=mock_send):
        await agent.respond(
            conversation=[Message(role="user", content="hello")],
            context=[],
            tools=[
                {
                    "name": "A",
                    "description": "a",
                    "input_schema": {},  # type: ignore[typeddict-item]
                },
                {
                    "name": "B",
                    "description": "b",
                    "input_schema": {},  # type: ignore[typeddict-item]
                },
            ],
        )

    sent_tools = mock_send.call_args.kwargs["tools"]
    assert "cache_control" not in sent_tools[0]
    assert sent_tools[-1]["cache_control"] == {"type": "ephemeral"}


async def test_respond_marks_last_message_as_cache_breakpoint(
    config: AgentConfig, text_response: MagicMock
) -> None:
    """The most recently appended conversation turn is marked as a cache
    breakpoint so the growing per-turn history reuses earlier reads.
    """
    agent = OpenrouterAgent(config)
    mock_send = AsyncMock(return_value=text_response)
    with patch.object(agent._client.chat, "send_async", new=mock_send):
        await agent.respond(
            conversation=[
                Message(role="user", content="first"),
                Message(role="assistant", content="second"),
            ],
            context=[],
            tools=[],
        )

    sent_messages = mock_send.call_args.kwargs["messages"]
    assert sent_messages[1]["content"] == "first"
    last_content = sent_messages[-1]["content"]
    assert isinstance(last_content, list)
    assert last_content[-1]["cache_control"] == {"type": "ephemeral"}


async def test_respond_sends_web_fetch_plugin_when_enabled(
    text_response: MagicMock,
) -> None:
    """web_fetch=True enables OpenRouter's web-fetch plugin."""
    config = AgentConfig(model="openai/gpt-4o", web_fetch=True)
    agent = OpenrouterAgent(config)
    mock_send = AsyncMock(return_value=text_response)
    with patch.object(agent._client.chat, "send_async", new=mock_send):
        await agent.respond(conversation=[], context=[], tools=[])

    sent_plugins = mock_send.call_args.kwargs["plugins"]
    assert sent_plugins == [{"id": "web-fetch"}]


async def test_respond_omits_web_fetch_plugin_by_default(
    config: AgentConfig, text_response: MagicMock
) -> None:
    """web_fetch=False (the default) sends no plugins."""
    agent = OpenrouterAgent(config)
    mock_send = AsyncMock(return_value=text_response)
    with patch.object(agent._client.chat, "send_async", new=mock_send):
        await agent.respond(conversation=[], context=[], tools=[])

    assert mock_send.call_args.kwargs["plugins"] is None


async def test_respond_reuses_system_prompt_for_same_context_object(
    config: AgentConfig, text_response: MagicMock
) -> None:
    """The system prompt is rebuilt once per turn (same context list
    object), not once per loop iteration.
    """
    agent = OpenrouterAgent(config)
    mock_send = AsyncMock(return_value=text_response)
    context = [_fragment()]

    with (
        patch.object(agent._client.chat, "send_async", new=mock_send),
        patch(FRAGMENT_SPY_TARGET, wraps=render_context_fragment) as spy,
    ):
        await agent.respond(conversation=[], context=context, tools=[])
        await agent.respond(conversation=[], context=context, tools=[])

    assert spy.call_count == 1
