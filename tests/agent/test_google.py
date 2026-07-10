"""Contract tests for GoogleAgent using a mocked HTTP client."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from agentsh.agent.google import GoogleAgent, _build_google_schema

from agentsh.config import AgentConfig
from agentsh.context.sanitize import render_context_fragment
from agentsh.models import ContextFragment, Message
from agentsh.tools import SchemaDict

FRAGMENT_SPY_TARGET = "agentsh.agent.google.render_context_fragment"
SCHEMA_SPY_TARGET = "agentsh.agent.google._build_google_schema"


def _fragment(provider: str = "git") -> ContextFragment:
    """Build a minimal context fragment for cache-identity tests."""
    return ContextFragment(provider=provider, summary="on main", payload={})


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


async def test_respond_reuses_system_instruction_for_same_context_object(
    config: AgentConfig, text_response: MagicMock
) -> None:
    """The system instruction is rebuilt once per turn (same context
    list object), not once per loop iteration -- Google's GenAI SDK has
    no per-request cache_control breakpoint like Anthropic's, so
    avoiding redundant Python-side reconstruction is this backend's
    caching optimization (see GoogleAgent.respond's docstring).
    """
    agent = GoogleAgent(config)
    mock_generate = AsyncMock(return_value=text_response)
    context = [_fragment()]

    with (
        patch.object(
            agent._client.aio.models, "generate_content", new=mock_generate
        ),
        patch(FRAGMENT_SPY_TARGET, wraps=render_context_fragment) as spy,
    ):
        await agent.respond(conversation=[], context=context, tools=[])
        await agent.respond(conversation=[], context=context, tools=[])

    assert spy.call_count == 1


async def test_respond_rebuilds_system_instruction_for_a_new_turn(
    config: AgentConfig, text_response: MagicMock
) -> None:
    """A new turn supplies a new context list object, so the cache
    correctly rebuilds rather than serving a stale system instruction.
    """
    agent = GoogleAgent(config)
    mock_generate = AsyncMock(return_value=text_response)

    with (
        patch.object(
            agent._client.aio.models, "generate_content", new=mock_generate
        ),
        patch(FRAGMENT_SPY_TARGET, wraps=render_context_fragment) as spy,
    ):
        await agent.respond(conversation=[], context=[_fragment()], tools=[])
        await agent.respond(conversation=[], context=[_fragment()], tools=[])

    assert spy.call_count == 2


async def test_respond_reuses_tool_declarations_for_same_tools_object(
    config: AgentConfig, text_response: MagicMock
) -> None:
    """The converted tool declarations are rebuilt once per turn (same
    tools list object), not once per loop iteration.
    """
    agent = GoogleAgent(config)
    mock_generate = AsyncMock(return_value=text_response)
    tools: list[SchemaDict] = [
        {
            "name": "RunCommand",
            "description": "run a command",
            "input_schema": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        }
    ]

    with (
        patch.object(
            agent._client.aio.models, "generate_content", new=mock_generate
        ),
        patch(SCHEMA_SPY_TARGET, wraps=_build_google_schema) as spy,
    ):
        await agent.respond(conversation=[], context=[], tools=tools)
        await agent.respond(conversation=[], context=[], tools=tools)

    assert spy.call_count == 1
