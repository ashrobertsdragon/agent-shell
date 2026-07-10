"""Anthropic Claude backend.

When `AgentConfig.web_fetch` is enabled, this backend registers
Anthropic's native server-side `web_fetch` tool on every request. That
tool runs entirely on Anthropic's infrastructure -- it never passes
through `PermissionEngine.evaluate()` or any other tool in
`agentsh.tools`, so enabling `web_fetch` intentionally bypasses the
permission engine for outbound web fetches. This is a documented
exception, not an oversight: see `docs/security.md` for the rationale.
"""

from typing import cast

import anthropic
from anthropic.types import (
    CacheControlEphemeralParam,
    MessageParam,
    TextBlockParam,
    ToolResultBlockParam,
    ToolUnionParam,
    ToolUseBlockParam,
    WebFetchTool20260318Param,
)

from agentsh.agent._system import _build_system
from agentsh.agent.base import Agent, register
from agentsh.agent.caching import IdentityCache
from agentsh.config import AgentConfig
from agentsh.models import ContextFragment, Message, ToolCall
from agentsh.tools import SchemaDict

_EPHEMERAL_CACHE: CacheControlEphemeralParam = {"type": "ephemeral"}
_WEB_FETCH_TOOL: WebFetchTool20260318Param = {
    "type": "web_fetch_20260318",
    "name": "web_fetch",
}


def _message_to_anthropic(m: Message, *, cache: bool = False) -> MessageParam:
    """Convert a canonical Message to Anthropic's message format.

    When `cache` is set, a cache breakpoint is added to the last content
    block so the next iteration of the same user turn (which resends
    this message unchanged, plus whatever was appended after it) can
    read this prefix from cache instead of reprocessing it.
    """
    if m.tool_results:
        results: list[ToolResultBlockParam] = [
            {
                "type": "tool_result",
                "tool_use_id": tr.call_id,
                "content": tr.content,
                "is_error": tr.is_error,
            }
            for tr in m.tool_results
        ]
        if cache and results:
            results[-1]["cache_control"] = _EPHEMERAL_CACHE  # type: ignore[typeddict-item]
        return {"role": "user", "content": results}

    content: list[ToolUseBlockParam | TextBlockParam] = []
    if m.content:
        content.append({"type": "text", "text": m.content})
    for tc in m.tool_calls:
        content.append(
            {
                "type": "tool_use",
                "id": tc.call_id,
                "name": tc.tool_name,
                "input": cast(dict[str, object], tc.arguments),
            }
        )

    if not cache and len(content) == 1 and content[0]["type"] == "text":
        return {"role": m.role, "content": m.content}  # type: ignore[typeddict-item]

    if cache and content:
        content[-1]["cache_control"] = _EPHEMERAL_CACHE  # type: ignore[typeddict-item]

    return {"role": m.role, "content": content}  # type: ignore[typeddict-item]


@register("anthropic")
class AnthropicAgent(Agent):
    """LLM backend using the Anthropic Messages API."""

    def __init__(self, config: AgentConfig) -> None:
        """Initialise the async Anthropic client and per-turn caches."""
        self._config = config
        self._client = anthropic.AsyncAnthropic()
        self._system_cache: IdentityCache[list[TextBlockParam]] = (
            IdentityCache()
        )
        self._tools_cache: IdentityCache[list[ToolUnionParam]] = IdentityCache()

    async def respond(
        self,
        conversation: list[Message],
        context: list[ContextFragment],
        tools: list[SchemaDict],
    ) -> Message:
        """Call the Anthropic API and return the next assistant message.

        `context` and `tools` are fixed for an entire user turn --
        `run_agent_loop` passes the same list objects on every iteration
        -- so the rendered system prompt and converted tool list are
        memoized by object identity rather than rebuilt on every one of
        up to 20 iterations.
        Cache breakpoints (`cache_control`) are placed on the last tool,
        the last system block, and the last message so Anthropic can
        serve the static prefix -- tools, system prompt, and the
        already-seen conversation history -- from cache instead of
        reprocessing it each iteration.

        When `self._config.web_fetch` is set, Anthropic's server-side
        `web_fetch` tool is appended after the caller's own tools. It
        executes on Anthropic's infrastructure and never reaches
        `agentsh.tools` or the permission engine -- see the module
        docstring.
        """

        def _build_tools() -> list[ToolUnionParam]:
            built: list[ToolUnionParam] = [
                {
                    "name": t["name"],
                    "description": t["description"],
                    "input_schema": {
                        "type": t.get("input_schema", {}).get("type"),
                        "properties": t.get("input_schema", {}).get(
                            "properties"
                        ),
                    },
                }
                for t in tools
            ]
            if self._config.web_fetch:
                built.append(_WEB_FETCH_TOOL.copy())
            if built:
                built[-1]["cache_control"] = _EPHEMERAL_CACHE  # type: ignore[typeddict-item]
            return built

        def _build_system_blocks() -> list[TextBlockParam]:
            return [
                {
                    "type": "text",
                    "text": _build_system(context),
                    "cache_control": _EPHEMERAL_CACHE,  # type: ignore[typeddict-item]
                }
            ]

        anthropic_tools = self._tools_cache.get_or_build(tools, _build_tools)
        system_blocks = self._system_cache.get_or_build(
            context, _build_system_blocks
        )

        messages = [
            _message_to_anthropic(m, cache=(i == len(conversation) - 1))
            for i, m in enumerate(conversation)
        ]

        response = await self._client.messages.create(
            model=self._config.model,
            max_tokens=self._config.max_tokens,
            system=system_blocks,
            messages=messages,
            tools=anthropic_tools,
        )

        tool_calls = tuple(
            ToolCall(
                tool_name=block.name,
                arguments=dict(block.input),  # type: ignore[arg-type]
                call_id=block.id,
            )
            for block in response.content
            if block.type == "tool_use"
        )

        text_content = " ".join(
            block.text  # type: ignore[union-attr]
            for block in response.content
            if block.type == "text"
        )

        return Message(
            role="assistant", content=text_content, tool_calls=tool_calls
        )
