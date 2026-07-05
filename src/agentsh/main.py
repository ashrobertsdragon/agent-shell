"""CLI entry point."""

import asyncio

from agentsh.agent import Agent
from agentsh.app import App, AppState
from agentsh.config import load_config
from agentsh.context import providers
from agentsh.context.builder import ContextBuilder
from agentsh.events import EventBus
from agentsh.permissions import PermissionEngine
from agentsh.repl import run_repl
from agentsh.shell import create_shell
from agentsh.tools.protocol import ToolRegistry
from agentsh.tools.read_file import ReadFile
from agentsh.tools.run_command import RunCommand
from agentsh.tools.write_file import WriteFile


def _build_app() -> App:
    """Wire together the runtime dependencies from config."""
    config = load_config()
    shell = create_shell(config.shell)
    permissions = PermissionEngine(config.permissions)

    tools = ToolRegistry()
    tools.register(RunCommand(shell=shell, permissions=permissions))
    tools.register(ReadFile())
    tools.register(WriteFile())

    context_builder = ContextBuilder(
        providers=[
            providers.GitProvider(),
            providers.FilesystemProvider(),
            providers.PythonProvider(),
            providers.DockerProvider(),
            providers.HistoryProvider(),
            providers.EnvironmentProvider(),
        ],
        timeout_ms=config.context.timeout_ms,
    )

    agent = Agent.from_provider(config.agent.provider)(config.agent)

    return App(
        shell=shell,
        tools=tools,
        permissions=permissions,
        context_builder=context_builder,
        agent=agent,
        state=AppState(),
        event_bus=EventBus(),
    )


def main() -> None:
    """Entry point for the agentsh CLI."""
    app = _build_app()
    asyncio.run(run_repl(app))


if __name__ == "__main__":
    main()
