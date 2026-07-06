"""Anthropic Claude backend."""

import json
from typing import cast

import anthropic
from anthropic.types import (
    MessageParam,
    TextBlockParam,
    ToolParam,
    ToolUseBlockParam,
)

from agentsh.agent import SYSTEM_PREFIX, Agent
from agentsh.config import AgentConfig
from agentsh.models import ContextFragment, Message, ToolCall
from agentsh.tools import SchemaDict


def _build_system(context: list[ContextFragment]) -> str:
    """Combine the base system prompt with serialized context fragments."""
    parts = [SYSTEM_PREFIX]
    for frag in context:
        parts.append(
            f"\n## {frag.summary}\n"
            f"```json\n{json.dumps(frag.payload, indent=2)}\n```"
        )
    return "\n".join(parts)


def _message_to_anthropic(m: Message) -> MessageParam:
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

    if len(content) == 1 and content[0]["type"] == "text":
        return {"role": m.role, "content": m.content}  # type: ignore[typeddict-item]
    return {"role": m.role, "content": content}  # type: ignore[typeddict-item]


class AnthropicAgent(Agent):
    """LLM backend using the Anthropic Messages API."""

    def __init__(self, config: AgentConfig) -> None:
        """Initialise the async Anthropic client."""
        self._config = config
        self._client = anthropic.AsyncAnthropic()

    async def respond(
        self,
        conversation: list[Message],
        context: list[ContextFragment],
        tools: list[SchemaDict],
    ) -> Message:
        """Call the Anthropic API and return the next assistant message."""
        anthropic_tools: list[ToolParam] = [
            {
                "name": t["name"],
                "description": t["description"],
                "input_schema": {
                    "type": t.get("input_schema", {}).get("type"),
                    "properties": t.get("input_schema", {}).get("properties"),
                },
            }
            for t in tools
        ]

        response = await self._client.messages.create(
            model=self._config.model,
            max_tokens=self._config.max_tokens,
            system=_build_system(context),
            messages=[_message_to_anthropic(m) for m in conversation],
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
