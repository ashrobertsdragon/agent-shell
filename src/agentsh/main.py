"""CLI entry point."""

from __future__ import annotations

import asyncio

from agentsh.app import App, AppState
from agentsh.repl import run_repl
from agentsh.shell.bash import BashShell
from agentsh.tools.protocol import ToolRegistry
from agentsh.tools.run_command import RunCommand


def _build_app() -> App:
    """Wire together the runtime dependencies."""
    shell = BashShell()
    tools = ToolRegistry()
    tools.register(RunCommand(shell=shell, permissions=None))
    return App(shell=shell, tools=tools, state=AppState())


def main() -> None:
    """Entry point for the agentsh CLI."""
    app = _build_app()
    asyncio.run(run_repl(app))
