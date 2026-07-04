"""CLI entry point."""

from __future__ import annotations

import asyncio

from agentsh.app import App, AppState
from agentsh.config import load_config
from agentsh.permissions import PermissionEngine
from agentsh.repl import run_repl
from agentsh.shell.bash import BashShell
from agentsh.tools.protocol import ToolRegistry
from agentsh.tools.run_command import RunCommand


def _build_app() -> App:
    """Wire together the runtime dependencies from config."""
    config = load_config()
    shell = BashShell()
    permissions = PermissionEngine(config.permissions)
    tools = ToolRegistry()
    tools.register(RunCommand(shell=shell, permissions=permissions))
    return App(shell=shell, tools=tools, permissions=permissions, state=AppState())


def main() -> None:
    """Entry point for the agentsh CLI."""
    app = _build_app()
    asyncio.run(run_repl(app))
