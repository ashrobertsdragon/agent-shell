"""CLI entry point."""

from __future__ import annotations

import asyncio

from agentsh.agent.anthropic import AnthropicAgent
from agentsh.agent.router import AgentRouter
from agentsh.app import App, AppState
from agentsh.config import load_config
from agentsh.context.builder import ContextBuilder
from agentsh.context.providers.docker import DockerProvider
from agentsh.context.providers.environment import EnvironmentProvider
from agentsh.context.providers.filesystem import FilesystemProvider
from agentsh.context.providers.git import GitProvider
from agentsh.context.providers.history import HistoryProvider
from agentsh.context.providers.kubernetes import KubernetesProvider
from agentsh.context.providers.python_env import PythonEnvProvider
from agentsh.events import EventBus
from agentsh.permissions import PermissionEngine
from agentsh.repl import run_repl
from agentsh.shell.bash import BashShell
from agentsh.tools.protocol import ToolRegistry
from agentsh.tools.read_file import ReadFile
from agentsh.tools.run_command import RunCommand
from agentsh.tools.write_file import WriteFile


def _build_app() -> App:
    """Wire together the runtime dependencies from config."""
    config = load_config()
    shell = BashShell()
    permissions = PermissionEngine(config.permissions)

    tools = ToolRegistry()
    tools.register(RunCommand(shell=shell, permissions=permissions))
    tools.register(ReadFile())
    tools.register(WriteFile())

    context_builder = ContextBuilder(
        providers=[
            GitProvider(),
            FilesystemProvider(),
            PythonEnvProvider(),
            DockerProvider(),
            KubernetesProvider(),
            HistoryProvider(),
            EnvironmentProvider(),
        ],
        timeout_ms=config.context.timeout_ms,
    )

    agents = {
        name: AnthropicAgent(backend_cfg)
        for name, backend_cfg in config.agent.backends.items()
    }
    agent_router = AgentRouter(config=config.agent, agents=agents)

    return App(
        shell=shell,
        tools=tools,
        permissions=permissions,
        context_builder=context_builder,
        agent_router=agent_router,
        state=AppState(),
        event_bus=EventBus(),
    )


def main() -> None:
    """Entry point for the agentsh CLI."""
    app = _build_app()
    asyncio.run(run_repl(app))
