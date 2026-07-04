"""REPL loop and UI helpers."""

from __future__ import annotations

import sys
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import ANSI
from prompt_toolkit.history import FileHistory

from agentsh.app import App
from agentsh.models import CommandResult, Message


def _render(result: CommandResult | Message) -> None:
    """Print a CommandResult or Message to stdout/stderr."""
    match result:
        case CommandResult():
            if result.stdout:
                print(result.stdout, end="")
            if result.stderr:
                print(result.stderr, end="", file=sys.stderr)
        case Message():
            if result.content:
                print(result.content)


async def run_repl(app: App) -> None:
    """Run the main REPL loop until EOF or KeyboardInterrupt."""
    history_dir = Path.home() / ".local" / "share" / "agentsh"
    history_dir.mkdir(parents=True, exist_ok=True)
    session: PromptSession[str] = PromptSession(
        history=FileHistory(str(history_dir / "history"))
    )

    while True:
        try:
            prompt = await app.shell.render_prompt()
            raw: str = await session.prompt_async(ANSI(prompt))
        except EOFError:
            break
        except KeyboardInterrupt:
            continue

        raw = raw.strip()
        if not raw:
            continue

        await app.shell.append_history(raw)
        result = await app.tools.get("RunCommand").invoke(command=raw)
        _render(result)
