"""OpenAI backend."""

import json

import openai
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionMessageToolCallParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionToolParam,
    ChatCompletionUserMessageParam,
)

from agentsh.agent import SYSTEM_PREFIX, Agent
from agentsh.config import AgentConfig
from agentsh.context.sanitize import render_context_fragment
from agentsh.models import ContextFragment, Message, ToolCall
from agentsh.tools import SchemaDict


def _build_system(context: list[ContextFragment]) -> str:
    """Combine the base system prompt with sanitized context fragments."""
    parts = [SYSTEM_PREFIX]
    parts.extend(render_context_fragment(frag) for frag in context)
    return "\n".join(parts)


def _message_to_openai(m: Message) -> list[ChatCompletionMessageParam]:
    """Convert a canonical Message to OpenAI's message format."""
    if m.tool_results:
        results: list[ChatCompletionMessageParam] = []
        for tr in m.tool_results:
            msg_tool: ChatCompletionToolMessageParam = {
                "role": "tool",
                "tool_call_id": tr.call_id,
                "content": tr.content,
            }
            results.append(msg_tool)
        return results

    if m.role == "user":
        msg_user: ChatCompletionUserMessageParam = {
            "role": "user",
            "content": m.content,
        }
        return [msg_user]

    if m.role == "assistant":
        tool_calls: list[ChatCompletionMessageToolCallParam] | None = None
        if m.tool_calls:
            tool_calls = []
            for tc in m.tool_calls:
                call: ChatCompletionMessageToolCallParam = {
                    "id": tc.call_id,
                    "type": "function",
                    "function": {
                        "name": tc.tool_name,
                        "arguments": json.dumps(tc.arguments),
                    },
                }
                tool_calls.append(call)

        if m.content and tool_calls is not None:
            msg_asst1: ChatCompletionAssistantMessageParam = {
                "role": "assistant",
                "content": m.content,
                "tool_calls": tool_calls,
            }
            return [msg_asst1]
        elif m.content:
            msg_asst2: ChatCompletionAssistantMessageParam = {
                "role": "assistant",
                "content": m.content,
            }
            return [msg_asst2]
        elif tool_calls is not None:
            msg_asst3: ChatCompletionAssistantMessageParam = {
                "role": "assistant",
                "tool_calls": tool_calls,
            }
            return [msg_asst3]

        msg_asst4: ChatCompletionAssistantMessageParam = {"role": "assistant"}
        return [msg_asst4]

    return []


class OpenaiAgent(Agent):
    """LLM backend using the OpenAI API."""

    def __init__(self, config: AgentConfig) -> None:
        """Initialise the async OpenAI client."""
        self._config = config
        self._client = openai.AsyncOpenAI()

    async def respond(
        self,
        conversation: list[Message],
        context: list[ContextFragment],
        tools: list[SchemaDict],
    ) -> Message:
        """Call the OpenAI API and return the next assistant message."""
        openai_tools: list[ChatCompletionToolParam] = []
        for t in tools:
            schema = dict(t.get("input_schema", {}))

            tool_param: ChatCompletionToolParam = {
                "type": "function",
                "function": {
                    "name": str(t["name"]),
                    "description": str(t.get("description", "")),
                    "parameters": schema,
                },
            }
            openai_tools.append(tool_param)

        messages: list[ChatCompletionMessageParam] = []
        sys_msg: ChatCompletionSystemMessageParam = {
            "role": "system",
            "content": _build_system(context),
        }
        messages.append(sys_msg)
        for m in conversation:
            messages.extend(_message_to_openai(m))

        if openai_tools:
            response = await self._client.chat.completions.create(
                model=self._config.model,
                messages=messages,
                tools=openai_tools,
                max_tokens=self._config.max_tokens,
            )
        else:
            response = await self._client.chat.completions.create(
                model=self._config.model,
                messages=messages,
                max_tokens=self._config.max_tokens,
            )

        choice = response.choices[0].message

        tool_calls: list[ToolCall] = []
        if choice.tool_calls:
            for tc in choice.tool_calls:
                if tc.type == "function":
                    try:
                        args = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        args = {}
                    if isinstance(args, dict):
                        tool_calls.append(
                            ToolCall(
                                tool_name=tc.function.name,
                                arguments=args,
                                call_id=tc.id,
                            )
                        )

        content = choice.content or ""
        if not isinstance(content, str):
            content = str(content)

        return Message(
            role="assistant",
            content=content,
            tool_calls=tuple(tool_calls),
        )
