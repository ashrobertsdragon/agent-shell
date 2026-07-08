"""The agentic tool-call loop — runs until the agent stops requesting tools."""

from typing import TYPE_CHECKING

from agentsh.events import AgentResponded, EventBus, ToolDenied, ToolInvoked
from agentsh.models import CommandResult, Message, ToolResult
from agentsh.permissions import (
    PermissionDeniedError,
    PermissionLevel,
    tool_call_key,
)

if TYPE_CHECKING:
    from agentsh.agent.base import Agent
    from agentsh.models import ContextFragment
    from agentsh.permissions import PermissionEngine
    from agentsh.repl import UI
    from agentsh.tools.protocol import ToolRegistry


class AgentLoopLimitError(Exception):
    """Raised when the agentic loop exceeds its maximum iteration count."""


async def run_agent_loop(
    *,
    agent: Agent,
    conversation: list[Message],
    context: list[ContextFragment],
    tools: ToolRegistry,
    permissions: PermissionEngine,
    ui: UI,
    event_bus: EventBus | None = None,
    max_iterations: int = 20,
) -> Message:
    """Run the agent until it produces a final response with no tool calls.

    DENY-level calls are short-circuited here so the agent gets a fast,
    clear rejection. CONFIRM-level calls are enforced inside each tool's
    own invoke() (which prompts via its injected confirm callback); a
    PermissionDeniedError raised from there is turned into an error
    ToolResult so the agent can recover gracefully.

    Raises AgentLoopLimitError if the loop exceeds max_iterations without
    a terminal (tool-call-free) response.
    """
    bus = event_bus or EventBus()

    for _iteration in range(max_iterations):
        response = await agent.respond(conversation, context, tools.schemas())
        conversation.append(response)

        await bus.publish(
            AgentResponded(
                content=response.content,
                tool_call_count=len(response.tool_calls),
            )
        )

        if not response.tool_calls:
            return response

        tool_results: list[ToolResult] = []
        for tc in response.tool_calls:
            key = tool_call_key(tc.tool_name, tc.arguments)
            if permissions.evaluate(key) == PermissionLevel.DENY:
                await bus.publish(ToolDenied(tool_name=tc.tool_name, key=key))
                tool_results.append(
                    ToolResult(
                        call_id=tc.call_id,
                        content="Permission denied by policy.",
                        is_error=True,
                    )
                )
                continue

            try:
                tool = tools.get(tc.tool_name)
                result: object = await tool.invoke(**tc.arguments)
                if isinstance(result, CommandResult):
                    content = (
                        f"stdout: {result.stdout}\n"
                        f"stderr: {result.stderr}\n"
                        f"exit_code: {result.exit_code}"
                    )
                else:
                    content = str(result)
                await bus.publish(
                    ToolInvoked(
                        tool_name=tc.tool_name,
                        arguments=dict(tc.arguments),
                        success=True,
                    )
                )
                tool_results.append(
                    ToolResult(call_id=tc.call_id, content=content)
                )
            except PermissionDeniedError as e:
                await bus.publish(ToolDenied(tool_name=tc.tool_name, key=key))
                tool_results.append(
                    ToolResult(
                        call_id=tc.call_id, content=str(e), is_error=True
                    )
                )
            except Exception as e:
                await bus.publish(
                    ToolInvoked(
                        tool_name=tc.tool_name,
                        arguments=dict(tc.arguments),
                        success=False,
                    )
                )
                tool_results.append(
                    ToolResult(
                        call_id=tc.call_id,
                        content=f"Error: {e}",
                        is_error=True,
                    )
                )

        conversation.append(
            Message(role="tool", content="", tool_results=tuple(tool_results))
        )

    raise AgentLoopLimitError(
        f"Agent loop exceeded {max_iterations} "
        "iterations without a terminal response."
    )
