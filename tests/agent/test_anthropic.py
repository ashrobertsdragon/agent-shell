"""Contract tests for AnthropicAgent using a mocked HTTP client."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from agentsh.agent.anthropic import AnthropicAgent

from agentsh.config import AgentConfig
from agentsh.context.sanitize import render_context_fragment
from agentsh.models import ContextFragment, Message

FRAGMENT_SPY_TARGET = "agentsh.agent.anthropic.render_context_fragment"


def _fragment(provider: str = "git") -> ContextFragment:
    """Build a minimal context fragment for cache-identity tests."""
    return ContextFragment(provider=provider, summary="on main", payload={})


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


async def test_respond_marks_system_prompt_as_cache_breakpoint(
    config: AgentConfig, text_response: MagicMock
) -> None:
    """The system prompt is sent as a block list with a cache breakpoint
    on its last block, so tools + system are reused across iterations.
    """
    agent = AnthropicAgent(config)
    mock_create = AsyncMock(return_value=text_response)
    with patch.object(agent._client.messages, "create", new=mock_create):
        await agent.respond(
            conversation=[Message(role="user", content="hello")],
            context=[],
            tools=[],
        )

    system = mock_create.call_args.kwargs["system"]
    assert isinstance(system, list)
    assert system[-1]["cache_control"] == {"type": "ephemeral"}


async def test_respond_marks_last_tool_as_cache_breakpoint(
    config: AgentConfig, text_response: MagicMock
) -> None:
    """Only the last tool definition carries a cache breakpoint."""
    agent = AnthropicAgent(config)
    mock_create = AsyncMock(return_value=text_response)
    with patch.object(agent._client.messages, "create", new=mock_create):
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

    sent_tools = mock_create.call_args.kwargs["tools"]
    assert "cache_control" not in sent_tools[0]
    assert sent_tools[-1]["cache_control"] == {"type": "ephemeral"}


async def test_respond_marks_last_message_as_cache_breakpoint(
    config: AgentConfig, text_response: MagicMock
) -> None:
    """The most recently appended conversation turn is marked as a cache
    breakpoint so the growing per-turn history reuses earlier reads.
    """
    agent = AnthropicAgent(config)
    mock_create = AsyncMock(return_value=text_response)
    with patch.object(agent._client.messages, "create", new=mock_create):
        await agent.respond(
            conversation=[
                Message(role="user", content="first"),
                Message(role="assistant", content="second"),
            ],
            context=[],
            tools=[],
        )

    sent_messages = mock_create.call_args.kwargs["messages"]
    assert sent_messages[0]["content"] == "first"
    last_content = sent_messages[-1]["content"]
    assert isinstance(last_content, list)
    assert last_content[-1]["cache_control"] == {"type": "ephemeral"}


async def test_respond_reuses_system_prompt_for_same_context_object(
    config: AgentConfig, text_response: MagicMock
) -> None:
    """The system prompt is rebuilt once per turn (same context list
    object), not once per loop iteration.
    """
    agent = AnthropicAgent(config)
    mock_create = AsyncMock(return_value=text_response)
    context = [_fragment()]

    with (
        patch.object(agent._client.messages, "create", new=mock_create),
        patch(FRAGMENT_SPY_TARGET, wraps=render_context_fragment) as spy,
    ):
        await agent.respond(conversation=[], context=context, tools=[])
        await agent.respond(conversation=[], context=context, tools=[])

    assert spy.call_count == 1


async def test_respond_rebuilds_system_prompt_for_a_new_turn(
    config: AgentConfig, text_response: MagicMock
) -> None:
    """A new turn supplies a new context list object, so the cache
    correctly rebuilds rather than serving a stale prompt.
    """
    agent = AnthropicAgent(config)
    mock_create = AsyncMock(return_value=text_response)

    with (
        patch.object(agent._client.messages, "create", new=mock_create),
        patch(FRAGMENT_SPY_TARGET, wraps=render_context_fragment) as spy,
    ):
        await agent.respond(conversation=[], context=[_fragment()], tools=[])
        await agent.respond(conversation=[], context=[_fragment()], tools=[])

    assert spy.call_count == 2
