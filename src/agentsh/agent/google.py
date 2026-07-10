"""Google GenAI backend."""

import uuid
from collections.abc import Mapping

from google import genai
from google.genai import types

from agentsh.agent import Agent, _build_system
from agentsh.agent.caching import IdentityCache
from agentsh.config import AgentConfig
from agentsh.models import ContextFragment, Message, ToolCall
from agentsh.tools import SchemaDict


def _get_genai_type(t: object) -> types.Type:
    """Map a JSON-schema type name to the matching GenAI schema type."""
    t_str = str(t).upper() if t else "OBJECT"
    match t_str:
        case "STRING":
            return types.Type.STRING
        case "INTEGER":
            return types.Type.INTEGER
        case "NUMBER":
            return types.Type.NUMBER
        case "BOOLEAN":
            return types.Type.BOOLEAN
        case "ARRAY":
            return types.Type.ARRAY
        case _:
            return types.Type.OBJECT


def _build_google_schema(input_schema: Mapping[str, object]) -> types.Schema:
    properties_raw = input_schema.get("properties")
    schema_properties: dict[str, types.Schema] = {}
    if isinstance(properties_raw, dict):
        for prop_name, prop_val in properties_raw.items():
            if isinstance(prop_val, dict):
                prop_type = prop_val.get("type")
                prop_desc = prop_val.get("description")
                schema_properties[prop_name] = types.Schema(
                    type=_get_genai_type(prop_type),
                    description=str(prop_desc) if prop_desc else "",
                )

    schema_type = input_schema.get("type")
    required_raw = input_schema.get("required")
    required_list: list[str] = []
    if isinstance(required_raw, list):
        for req in required_raw:
            required_list.append(str(req))

    return types.Schema(
        type=_get_genai_type(schema_type),
        properties=schema_properties,
        required=required_list,
    )


def _message_to_google(
    m: Message, call_id_to_name: dict[str, str]
) -> types.Content:
    """Convert a canonical Message to Google GenAI's content format."""
    if m.tool_results:
        result_parts = []
        for tr in m.tool_results:
            tool_name = call_id_to_name.get(tr.call_id, "unknown_tool")
            response_dict = (
                {"output": tr.content}
                if not tr.is_error
                else {"error": tr.content}
            )
            part = types.Part.from_function_response(
                name=tool_name,
                response=response_dict,
            )
            if part.function_response:
                part.function_response.id = tr.call_id
            result_parts.append(part)
        return types.Content(role="user", parts=result_parts)

    out_parts: list[types.Part] = []
    if m.content:
        out_parts.append(types.Part.from_text(text=m.content))
    for tc in m.tool_calls:
        part = types.Part.from_function_call(
            name=tc.tool_name,
            args=dict(tc.arguments),
        )
        if part.function_call:
            part.function_call.id = tc.call_id
        out_parts.append(part)

    role = "model" if m.role == "assistant" else "user"
    return types.Content(role=role, parts=out_parts)


class GoogleAgent(Agent):
    """LLM backend using the Google GenAI API."""

    def __init__(self, config: AgentConfig) -> None:
        """Initialise the async Google GenAI client and per-turn caches."""
        self._config = config
        self._client = genai.Client()
        self._system_cache: IdentityCache[str] = IdentityCache()
        self._tools_cache: IdentityCache[
            list[types.Tool | object | types.FunctionDeclaration]
        ] = IdentityCache()

    async def respond(
        self,
        conversation: list[Message],
        context: list[ContextFragment],
        tools: list[SchemaDict],
    ) -> Message:
        """Call the Google GenAI API and return the next assistant message.

        `context` and `tools` are fixed for an entire user turn --
        `run_agent_loop` passes the same list objects on every
        iteration -- so the system instruction and converted tool
        declarations are memoized by object identity rather than
        rebuilt on every one of up to 20 iterations.

        Google's GenAI SDK has no per-request `cache_control` breakpoint
        like Anthropic's; server-side caching there is either fully
        automatic ("implicit" caching, which needs no code and simply
        benefits from a stable prefix) or requires explicitly creating
        and managing a `CachedContent` resource (`client.aio.caches`)
        with its own minimum-token threshold and TTL lifecycle. This
        agent's system prompt is small and unlikely to reliably clear
        that threshold, and the lifecycle overhead (create, refresh,
        invalidate on context change, delete) is not worth it for
        uncertain savings, so this backend only eliminates the
        redundant Python-side rebuild every iteration and otherwise
        relies on implicit caching for any server-side benefit.
        """

        def _build_tools() -> list[
            types.Tool | object | types.FunctionDeclaration
        ]:
            built: list[types.Tool | object | types.FunctionDeclaration] = []
            for t in tools:
                schema = _build_google_schema(t["input_schema"])
                tool = types.Tool(
                    function_declarations=[
                        types.FunctionDeclaration(
                            name=str(t["name"]),
                            description=str(t.get("description", "")),
                            parameters=schema,
                        )
                    ]
                )
                built.append(tool)
            return built

        google_tools = self._tools_cache.get_or_build(tools, _build_tools)
        system_instruction = self._system_cache.get_or_build(
            context, lambda: _build_system(context)
        )

        call_id_to_name: dict[str, str] = {}
        for m in conversation:
            for tc in m.tool_calls:
                call_id_to_name[tc.call_id] = tc.tool_name

        contents: list[types.Content] = []
        for m in conversation:
            content = _message_to_google(m, call_id_to_name)
            contents.append(content)

        if google_tools:
            generate_config = types.GenerateContentConfig(
                system_instruction=system_instruction,
                tools=google_tools,
                max_output_tokens=self._config.max_tokens,
            )
        else:
            generate_config = types.GenerateContentConfig(
                system_instruction=system_instruction,
                max_output_tokens=self._config.max_tokens,
            )

        response = await self._client.aio.models.generate_content(
            model=self._config.model,
            contents=contents,
            config=generate_config,
        )

        if not response.candidates:
            return Message(role="assistant", content="")

        choice = response.candidates[0].content

        text_parts = []
        tool_calls = []
        if choice and choice.parts:
            for part in choice.parts:
                if part.text:
                    text_parts.append(part.text)
                elif part.function_call:
                    call_id = part.function_call.id or str(uuid.uuid4())
                    args = (
                        dict(part.function_call.args)
                        if part.function_call.args
                        else {}
                    )
                    tool_calls.append(
                        ToolCall(
                            tool_name=part.function_call.name or "unknown",
                            arguments=args,
                            call_id=call_id,
                        )
                    )

        return Message(
            role="assistant",
            content=" ".join(text_parts),
            tool_calls=tuple(tool_calls),
        )
