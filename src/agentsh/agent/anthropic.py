"""Anthropic Claude backend."""

from __future__ import annotations

import json
from typing import Any, cast

import anthropic

from agentsh.config import AgentBackendConfig
from agentsh.models import ContextFragment, Message, ToolCall

_SYSTEM_PREFIX = (
    "You are an AI assistant integrated into the user's shell. "
    "Use the provided tools to help with tasks. "
    "Be concise — you are running inside a terminal."
)


def _build_system(context: list[ContextFragment]) -> str:
    """Combine the base system prompt with serialized context fragments."""
    parts = [_SYSTEM_PREFIX]
    for frag in context:
        parts.append(
            f"\n## {frag.summary}\n```json\n{json.dumps(frag.payload, indent=2)}\n```"
        )
    return "\n".join(parts)


def _message_to_anthropic(m: Message) -> dict[str, Any]:
    """Convert a canonical Message to Anthropic's message format."""
    if m.tool_results:
        return {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tr.call_id,
                    "content": tr.content,
                    "is_error": tr.is_error,
                }
                for tr in m.tool_results
            ],
        }

    content: list[dict[str, Any]] = []
    if m.content:
        content.append({"type": "text", "text": m.content})
    for tc in m.tool_calls:
        content.append(
            {
                "type": "tool_use",
                "id": tc.call_id,
                "name": tc.tool_name,
                "input": tc.arguments,
            }
        )

    if len(content) == 1 and content[0]["type"] == "text":
        return {"role": m.role, "content": m.content}
    return {"role": m.role, "content": content}


class AnthropicAgent:
    """LLM backend using the Anthropic Messages API."""

    def __init__(self, config: AgentBackendConfig) -> None:
        """Initialise the async Anthropic client."""
        self._config = config
        self._client = anthropic.AsyncAnthropic()

    async def respond(
        self,
        conversation: list[Message],
        context: list[ContextFragment],
        tools: list[dict[str, Any]],
    ) -> Message:
        """Call the Anthropic API and return the next assistant message."""
        anthropic_tools = [
            {
                "name": t["name"],
                "description": t["description"],
                "input_schema": t.get(
                    "input_schema", {"type": "object", "properties": {}}
                ),
            }
            for t in tools
        ]

        response = await self._client.messages.create(
            model=self._config.model,
            max_tokens=4096,
            system=_build_system(context),
            messages=cast(Any, [_message_to_anthropic(m) for m in conversation]),
            tools=cast(Any, anthropic_tools),
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

        return Message(role="assistant", content=text_content, tool_calls=tool_calls)
