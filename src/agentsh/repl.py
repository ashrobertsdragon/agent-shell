"""REPL loop and UI helpers."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import ANSI
from prompt_toolkit.history import FileHistory

from agentsh.app import App
from agentsh.models import CommandResult, Message


class UI:
    """Handles user-facing I/O: prompting, rendering results, and confirmations."""

    def __init__(self, session: PromptSession[str]) -> None:
        """Bind to an existing prompt_toolkit session."""
        self._session = session

    def render(self, result: CommandResult | Message) -> None:
        """Print a result to stdout (or stderr for command stderr output)."""
        match result:
            case CommandResult():
                if result.stdout:
                    print(result.stdout, end="")
                if result.stderr:
                    print(result.stderr, end="", file=sys.stderr)
            case Message():
                if result.content:
                    print(result.content)

    async def confirm(self, tool_name: str, arguments: dict[str, Any]) -> bool:
        """Prompt the user to allow or deny a CONFIRM-level tool call."""
        label = arguments.get("command") or arguments.get("path") or tool_name
        print(f"\n[agentsh] permission required — {tool_name}: {label}")
        try:
            answer = await self._session.prompt_async("Allow? [y/N] ")
            return answer.strip().lower() == "y"
        except (EOFError, KeyboardInterrupt):
            return False


async def run_repl(app: App) -> None:
    """Run the main REPL loop until EOF or KeyboardInterrupt."""
    from agentsh.permissions import PermissionLevel
    from agentsh.tools.run_command import PermissionDeniedError

    history_dir = Path.home() / ".local" / "share" / "agentsh"
    history_dir.mkdir(parents=True, exist_ok=True)
    session: PromptSession[str] = PromptSession(
        history=FileHistory(str(history_dir / "history"))
    )
    ui = UI(session)
    app.ui = ui

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

        try:
            run_cmd = app.tools.get("RunCommand")
            key = f"RunCommand:{raw}"
            level = app.permissions.evaluate(key)
            if level == PermissionLevel.DENY:
                print(f"[agentsh] denied: {raw}", file=sys.stderr)
                continue
            if level == PermissionLevel.CONFIRM and not await ui.confirm(
                "RunCommand", {"command": raw}
            ):
                print("[agentsh] cancelled.", file=sys.stderr)
                continue
            result = await run_cmd.invoke(command=raw)
            ui.render(result)
        except PermissionDeniedError as e:
            print(f"[agentsh] {e}", file=sys.stderr)
