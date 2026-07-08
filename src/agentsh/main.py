"""CLI entry point."""

import asyncio
import sys
from collections.abc import Mapping

from agentsh.agent import Agent
from agentsh.app import App, AppState
from agentsh.config import load_config
from agentsh.context.builder import ContextBuilder
from agentsh.context.providers import UnknownProviderError, build_providers
from agentsh.events import EventBus
from agentsh.models import JsonValue
from agentsh.permissions import PermissionEngine
from agentsh.repl import run_repl
from agentsh.shell import UnsupportedShellError, create_shell
from agentsh.tools.protocol import ToolRegistry
from agentsh.tools.read_file import ReadFile
from agentsh.tools.run_command import RunCommand
from agentsh.tools.write_file import WriteFile


def _build_app() -> App:
    """Wire together the runtime dependencies from config."""
    config = load_config()
    shell = create_shell(config.shell)
    permissions = PermissionEngine(config.permissions)

    context_builder = ContextBuilder(
        providers=build_providers(config.context.providers),
        timeout_ms=config.context.timeout_ms,
    )

    agent = Agent.from_provider(config.agent.provider)(config.agent)

    app = App(
        shell=shell,
        tools=ToolRegistry(),
        permissions=permissions,
        context_builder=context_builder,
        agent=agent,
        state=AppState(),
        event_bus=EventBus(),
    )

    async def confirm(
        tool_name: str, arguments: Mapping[str, JsonValue]
    ) -> bool:
        """Delegate CONFIRM prompts to the REPL's UI once it is attached.

        Tools are constructed before the UI exists, so this closes over
        app and reads app.ui lazily; by the time any tool is invoked,
        run_repl has already set it.
        """
        if app.ui is None:
            return False
        return await app.ui.confirm(tool_name, arguments)

    app.tools.register(
        RunCommand(shell=shell, permissions=permissions, confirm=confirm)
    )
    app.tools.register(ReadFile(permissions=permissions, confirm=confirm))
    app.tools.register(WriteFile(permissions=permissions, confirm=confirm))

    return app


def main() -> None:
    """Entry point for the agentsh CLI."""
    try:
        app = _build_app()
    except (UnsupportedShellError, UnknownProviderError) as e:
        sys.exit(f"agentsh: {e}")
    asyncio.run(run_repl(app))


if __name__ == "__main__":
    main()
