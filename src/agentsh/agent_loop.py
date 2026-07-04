"""The agentic tool-call loop — runs until the agent stops requesting tools."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agentsh.models import Message, ToolCall, ToolResult
from agentsh.permissions import PermissionLevel
from agentsh.tools.run_command import PermissionDeniedError

if TYPE_CHECKING:
    from agentsh.agent.protocol import Agent
    from agentsh.models import ContextFragment
    from agentsh.permissions import PermissionEngine
    from agentsh.repl import UI
    from agentsh.tools.protocol import ToolRegistry


def _tool_call_key(tc: ToolCall) -> str:
    """Build the permission key for a tool call."""
    match tc.tool_name:
        case "RunCommand":
            return f"RunCommand:{tc.arguments.get('command', '')}"
        case _:
            return tc.tool_name


async def run_agent_loop(
    *,
    agent: Agent,
    conversation: list[Message],
    context: list[ContextFragment],
    tools: ToolRegistry,
    permissions: PermissionEngine,
    ui: UI,
) -> Message:
    """Run the agent until it produces a final response with no tool calls.

    CONFIRM-level tool calls prompt the user; if denied, an error ToolResult
    is injected so the agent can recover gracefully.
    """
    while True:
        response = await agent.respond(conversation, context, tools.schemas())
        conversation.append(response)

        if not response.tool_calls:
            return response

        tool_results: list[ToolResult] = []
        for tc in response.tool_calls:
            key = _tool_call_key(tc)
            level = permissions.evaluate(key)

            match level:
                case PermissionLevel.DENY:
                    tool_results.append(
                        ToolResult(
                            call_id=tc.call_id,
                            content="Permission denied by policy.",
                            is_error=True,
                        )
                    )
                    continue
                case PermissionLevel.CONFIRM:
                    if not await ui.confirm(tc.tool_name, tc.arguments):
                        tool_results.append(
                            ToolResult(
                                call_id=tc.call_id,
                                content="Permission denied by user.",
                                is_error=True,
                            )
                        )
                        continue

            try:
                tool = tools.get(tc.tool_name)
                result: Any = await tool.invoke(**tc.arguments)
                content = (
                    (
                        f"stdout: {result.stdout}\n"
                        f"stderr: {result.stderr}\n"
                        f"exit_code: {result.exit_code}"
                    )
                    if hasattr(result, "stdout")
                    else str(result)
                )
                tool_results.append(ToolResult(call_id=tc.call_id, content=content))
            except PermissionDeniedError as e:
                tool_results.append(
                    ToolResult(call_id=tc.call_id, content=str(e), is_error=True)
                )
            except Exception as e:
                tool_results.append(
                    ToolResult(call_id=tc.call_id, content=f"Error: {e}", is_error=True)
                )

        conversation.append(
            Message(role="tool", content="", tool_results=tuple(tool_results))
        )
