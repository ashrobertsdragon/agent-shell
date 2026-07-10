"""OpenRouter backend."""

import json

from openrouter import OpenRouter
from openrouter.components import (
    ChatAssistantMessageTypedDict,
    ChatContentCacheControlTypedDict,
    ChatFunctionToolTypedDict,
    ChatSystemMessageTypedDict,
    ChatToolCallFunctionTypedDict,
    ChatToolCallTypedDict,
    ChatToolMessageTypedDict,
    ChatUserMessageTypedDict,
)

from agentsh.agent import SYSTEM_PREFIX, Agent
from agentsh.agent.caching import IdentityCache
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

_EPHEMERAL_CACHE: ChatContentCacheControlTypedDict = {"type": "ephemeral"}


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

        msg_asst: ChatAssistantMessageTypedDict
        match tool_calls, m.content:
            case list() as calls, content if content:
                msg_asst = {
                    "role": "assistant",
                    "content": content,
                    "tool_calls": calls,
                }
            case None, content if content:
                msg_asst = {"role": "assistant", "content": content}
            case list() as calls, _:
                msg_asst = {"role": "assistant", "tool_calls": calls}
            case _:
                msg_asst = {"role": "assistant"}
        return [msg_asst]

    return []


def _mark_cache_breakpoint(
    msg: OpenRouterMessageDict,
) -> OpenRouterMessageDict:
    """Return a copy of `msg` with its content wrapped for a cache mark.

    Only plain string content is convertible this way; a message whose
    content is empty or already structured is returned unchanged (no
    breakpoint), which is a safe no-op for the case actually reachable
    here -- run_agent_loop always ends a turn's conversation on a plain
    user or tool-result message before calling respond() again.
    """
    content = msg.get("content")
    if not isinstance(content, str) or not content:
        return msg
    marked = dict(msg)
    marked["content"] = [
        {"type": "text", "text": content, "cache_control": _EPHEMERAL_CACHE}
    ]
    return marked  # type: ignore[return-value]


class OpenrouterAgent(Agent):
    """LLM backend using the OpenRouter API."""

    def __init__(self, config: AgentConfig) -> None:
        """Initialise the async OpenRouter client and per-turn caches."""
        self._config = config
        self._client = OpenRouter()
        self._system_cache: IdentityCache[ChatSystemMessageTypedDict] = (
            IdentityCache()
        )
        self._tools_cache: IdentityCache[list[ChatFunctionToolTypedDict]] = (
            IdentityCache()
        )

    async def respond(
        self,
        conversation: list[Message],
        context: list[ContextFragment],
        tools: list[SchemaDict],
    ) -> Message:
        """Call the OpenRouter API and return the next assistant message.

        `context` and `tools` are fixed for an entire user turn --
        `run_agent_loop` passes the same list objects on every
        iteration -- so the system message and converted tool list are
        memoized by object identity rather than rebuilt on every one of
        up to 20 iterations. Cache breakpoints (`cache_control`) are
        placed on the last tool, the system message, and the last
        message so upstream providers that honor OpenRouter's
        Anthropic-style cache_control field (notably Anthropic models
        routed through OpenRouter) can serve the static prefix from
        cache instead of reprocessing it each iteration.
        """

        def _build_tools() -> list[ChatFunctionToolTypedDict]:
            built: list[ChatFunctionToolTypedDict] = []
            for t in tools:
                func: ChatFunctionToolTypedDict = {
                    "type": "function",
                    "function": {
                        "name": str(t["name"]),
                        "description": str(t.get("description", "")),
                        "parameters": dict(t["input_schema"]),
                    },
                }
                built.append(func)
            if built:
                built[-1]["cache_control"] = _EPHEMERAL_CACHE  # type: ignore[typeddict-item]
            return built

        def _build_system_message() -> ChatSystemMessageTypedDict:
            return {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": _build_system(context),
                        "cache_control": _EPHEMERAL_CACHE,
                    }
                ],
            }

        or_tools = self._tools_cache.get_or_build(tools, _build_tools)
        sys_msg = self._system_cache.get_or_build(
            context, _build_system_message
        )

        messages: list[OpenRouterMessageDict] = [sys_msg]
        for m in conversation:
            messages.extend(_message_to_openrouter(m))

        if len(messages) > 1:
            messages[-1] = _mark_cache_breakpoint(messages[-1])

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
