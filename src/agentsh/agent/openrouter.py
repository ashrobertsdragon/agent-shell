"""OpenRouter backend."""

import json

from openrouter import OpenRouter
from openrouter.components import (
    ChatAssistantMessageTypedDict,
    ChatFunctionToolTypedDict,
    ChatSystemMessageTypedDict,
    ChatToolCallFunctionTypedDict,
    ChatToolCallTypedDict,
    ChatToolMessageTypedDict,
    ChatUserMessageTypedDict,
)

from agentsh.agent import SYSTEM_PREFIX, Agent
from agentsh.config import AgentConfig
from agentsh.context.sanitize import render_context_fragment
from agentsh.models import ContextFragment, Message, ToolCall
from agentsh.tools import SchemaDict

type OpenRouterMessageDict = (
    ChatSystemMessageTypedDict
    | ChatUserMessageTypedDict
    | ChatAssistantMessageTypedDict
    | ChatToolMessageTypedDict
)


def _build_system(context: list[ContextFragment]) -> str:
    """Combine the base system prompt with sanitized context fragments."""
    parts = [SYSTEM_PREFIX]
    parts.extend(render_context_fragment(frag) for frag in context)
    return "\n".join(parts)


def _message_to_openrouter(m: Message) -> list[OpenRouterMessageDict]:
    """Convert a canonical Message to OpenRouter's message format."""
    if m.tool_results:
        tool_results: list[OpenRouterMessageDict] = []
        for tr in m.tool_results:
            msg_tool: ChatToolMessageTypedDict = {
                "role": "tool",
                "tool_call_id": tr.call_id,
                "content": tr.content,
            }
            tool_results.append(msg_tool)
        return tool_results

    if m.role == "user":
        msg_user: ChatUserMessageTypedDict = {
            "role": "user",
            "content": m.content,
        }
        return [msg_user]

    if m.role == "assistant":
        tool_calls: list[ChatToolCallTypedDict] | None = None
        if m.tool_calls:
            tool_calls = []
            for tc in m.tool_calls:
                func_dict: ChatToolCallFunctionTypedDict = {
                    "name": tc.tool_name,
                    "arguments": json.dumps(tc.arguments),
                }
                call: ChatToolCallTypedDict = {
                    "id": tc.call_id,
                    "type": "function",
                    "function": func_dict,
                }
                tool_calls.append(call)

        if m.content and tool_calls is not None:
            msg_asst1: ChatAssistantMessageTypedDict = {
                "role": "assistant",
                "content": m.content,
                "tool_calls": tool_calls,
            }
            return [msg_asst1]
        elif m.content:
            msg_asst2: ChatAssistantMessageTypedDict = {
                "role": "assistant",
                "content": m.content,
            }
            return [msg_asst2]
        elif tool_calls is not None:
            msg_asst3: ChatAssistantMessageTypedDict = {
                "role": "assistant",
                "tool_calls": tool_calls,
            }
            return [msg_asst3]

        msg_asst4: ChatAssistantMessageTypedDict = {"role": "assistant"}
        return [msg_asst4]

    return []


class OpenrouterAgent(Agent):
    """LLM backend using the OpenRouter API."""

    def __init__(self, config: AgentConfig) -> None:
        """Initialise the async OpenRouter client."""
        self._config = config
        self._client = OpenRouter()

    async def respond(
        self,
        conversation: list[Message],
        context: list[ContextFragment],
        tools: list[SchemaDict],
    ) -> Message:
        """Call the OpenRouter API and return the next assistant message."""
        or_tools: list[ChatFunctionToolTypedDict] = []
        for t in tools:
            func: ChatFunctionToolTypedDict = {
                "type": "function",
                "function": {
                    "name": str(t["name"]),
                    "description": str(t.get("description", "")),
                    "parameters": dict(t["input_schema"]),
                },
            }
            or_tools.append(func)

        messages: list[OpenRouterMessageDict] = []
        sys_msg: ChatSystemMessageTypedDict = {
            "role": "system",
            "content": _build_system(context),
        }
        messages.append(sys_msg)
        for m in conversation:
            messages.extend(_message_to_openrouter(m))

        if or_tools:
            response = await self._client.chat.send_async(
                model=self._config.model,
                messages=messages,
                tools=or_tools,
                max_tokens=self._config.max_tokens,
            )
        else:
            response = await self._client.chat.send_async(
                model=self._config.model,
                messages=messages,
                max_tokens=self._config.max_tokens,
            )

        choice = response.choices[0].message

        tool_calls: list[ToolCall] = []
        tool_calls_attr = getattr(choice, "tool_calls", None)
        if tool_calls_attr:
            for tc in tool_calls_attr:
                if getattr(tc, "type", "") == "function":
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
