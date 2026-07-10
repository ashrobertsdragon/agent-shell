"""OpenAI backend."""

import json
import uuid

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

from agentsh.agent import Agent, _build_system
from agentsh.agent.caching import IdentityCache
from agentsh.config import AgentConfig
from agentsh.models import ContextFragment, Message, ToolCall
from agentsh.tools import SchemaDict


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

        msg_asst: ChatCompletionAssistantMessageParam
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


class OpenaiAgent(Agent):
    """LLM backend using the OpenAI API."""

    def __init__(self, config: AgentConfig) -> None:
        """Initialise the async OpenAI client and per-turn caches.

        `_prompt_cache_key` is a stable identifier for this agent
        instance's lifetime (one shell session). OpenAI caches
        identical-prefix requests automatically, but routes them more
        reliably to the same cache-holding backend when a
        `prompt_cache_key` is supplied -- see
        https://platform.openai.com/docs/guides/prompt-caching.
        """
        self._config = config
        self._client = openai.AsyncOpenAI()
        self._prompt_cache_key = uuid.uuid4().hex
        self._system_cache: IdentityCache[ChatCompletionSystemMessageParam] = (
            IdentityCache()
        )
        self._tools_cache: IdentityCache[list[ChatCompletionToolParam]] = (
            IdentityCache()
        )

    async def respond(
        self,
        conversation: list[Message],
        context: list[ContextFragment],
        tools: list[SchemaDict],
    ) -> Message:
        """Call the OpenAI API and return the next assistant message.

        `context` and `tools` are fixed for an entire user turn --
        `run_agent_loop` passes the same list objects on every
        iteration -- so the system message and converted tool list are
        memoized by object identity rather than rebuilt on every one of
        up to 20 iterations.
        """

        def _build_tools() -> list[ChatCompletionToolParam]:
            built: list[ChatCompletionToolParam] = []
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
                built.append(tool_param)
            return built

        def _build_system_message() -> ChatCompletionSystemMessageParam:
            return {"role": "system", "content": _build_system(context)}

        openai_tools = self._tools_cache.get_or_build(tools, _build_tools)
        sys_msg = self._system_cache.get_or_build(
            context, _build_system_message
        )

        messages: list[ChatCompletionMessageParam] = [sys_msg]
        for m in conversation:
            messages.extend(_message_to_openai(m))

        if openai_tools:
            response = await self._client.chat.completions.create(
                model=self._config.model,
                messages=messages,
                tools=openai_tools,
                max_tokens=self._config.max_tokens,
                prompt_cache_key=self._prompt_cache_key,
            )
        else:
            response = await self._client.chat.completions.create(
                model=self._config.model,
                messages=messages,
                max_tokens=self._config.max_tokens,
                prompt_cache_key=self._prompt_cache_key,
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
