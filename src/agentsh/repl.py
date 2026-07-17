"""REPL loop and UI helpers."""

from __future__ import annotations

import sys
import time
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import ANSI
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.key_binding.key_processor import KeyPressEvent

from agentsh.agent_loop import AgentLoopLimitError, run_agent_loop
from agentsh.app import App
from agentsh.classifier import InputKind, agent_query, classify
from agentsh.events import CommandFinished, CommandStarted, ContextCollected
from agentsh.history_security import ensure_secure_file
from agentsh.limits import truncate_text
from agentsh.models import CommandResult, JsonValue, Message

_EXIT_KEYWORDS = frozenset({"exit", "quit"})

_PREVIEW_MAX_CHARS = 2000


def _sanitize_for_terminal(text: str) -> str:
    r"""Escape control sequences that could spoof or manipulate the terminal.

    Untrusted file content shown in a CONFIRM prompt must not be able to
    inject ANSI escape sequences (\x1b) or lone carriage returns (\r)
    that overwrite or disguise the prompt the user is approving.
    """
    return text.replace("\x1b", "\\x1b").replace("\r", "\\r")


def _content_preview(arguments: Mapping[str, JsonValue]) -> str | None:
    """Return a preview of the file content or patch a call would write.

    WriteFile calls carry the full content (or a SEARCH/REPLACE patch) in
    their arguments; surfacing it here means CONFIRM prompts show what
    will actually change, not just the target path, so approval isn't
    blind (issue #21). Returns None for calls with nothing to preview
    (e.g. RunCommand). An empty content/patch is still previewed
    explicitly (as "(empty)") since it may carry real meaning (e.g.
    truncating a file), not just skipped as if there were no payload.
    """
    for key in ("content", "patch"):
        value = arguments.get(key)
        if not isinstance(value, str):
            continue
        if not value:
            return f"--- {key} preview ---\n(empty)"
        sanitized = _sanitize_for_terminal(value)
        truncated = sanitized[:_PREVIEW_MAX_CHARS]
        suffix = (
            "\n... (truncated)" if len(sanitized) > _PREVIEW_MAX_CHARS else ""
        )
        return f"--- {key} preview ---\n{truncated}{suffix}"
    return None


class UI:
    """Handles user-facing I/O: prompts, rendering, and confirmations."""

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

    async def confirm(
        self, tool_name: str, arguments: Mapping[str, JsonValue]
    ) -> bool:
        """Prompt the user to allow or deny a CONFIRM-level tool call."""
        label = arguments.get("command") or arguments.get("path") or tool_name
        print(f"\n[agentsh] permission required — {tool_name}: {label}")
        preview = _content_preview(arguments)
        if preview is not None:
            print(preview)
        try:
            answer = await self._session.prompt_async("Allow? [y/N] ")
            return answer.strip().lower() == "y"
        except (EOFError, KeyboardInterrupt):
            return False


def _handle_ctrl_c(event: KeyPressEvent) -> None:
    """Clear the current input line, or quit agentsh if the prompt is empty.

    This mirrors a real shell: Ctrl+C on a partially typed line discards
    that line without leaving the REPL, but Ctrl+C at an empty prompt is
    an explicit exit (alongside ``exit``/``quit`` and Ctrl+D), so there is
    always an obvious way out. Exiting is signalled by making
    ``prompt_async`` raise ``EOFError``, which the loop already treats as
    a clean break.
    """
    if event.current_buffer.text:
        event.current_buffer.reset()
    else:
        event.app.exit(exception=EOFError())


def _build_key_bindings() -> KeyBindings:
    """Return the REPL's prompt key bindings (currently just Ctrl+C)."""
    bindings = KeyBindings()
    bindings.add("c-c")(_handle_ctrl_c)
    return bindings


async def run_repl(app: App) -> None:
    """Run the main REPL loop until EOF or KeyboardInterrupt."""
    history_dir = Path.home() / ".local" / "share" / "agentsh"
    history_path = history_dir / "history"
    ensure_secure_file(history_path)
    session: PromptSession[str] = PromptSession(
        history=FileHistory(str(history_path)),
        key_bindings=_build_key_bindings(),
    )
    ui = UI(session)
    app.ui = ui

    try:
        await _repl_loop(app, session, ui)
    finally:
        await app.shell.close()


async def _repl_loop(
    app: App,
    session: PromptSession[str],
    ui: UI,
) -> None:
    """Drive the prompt/classify/dispatch loop until the user exits.

    User-typed shell input is executed directly on the shell backend and
    is deliberately not routed through the permission engine: the engine
    exists to constrain the agent's tool calls, not the human at the
    keyboard. Only the AGENT path runs through permission enforcement
    (inside each tool's own invoke()).
    """
    bus = app.event_bus
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
        if raw in _EXIT_KEYWORDS:
            break

        await app.shell.append_history(raw)
        kind = await classify(raw, app.shell)

        match kind:
            case InputKind.SHELL:
                await bus.publish(CommandStarted(command=raw))
                t0 = time.monotonic()
                result = await app.shell.execute(raw)
                result = replace(
                    result,
                    stdout=truncate_text(result.stdout),
                    stderr=truncate_text(result.stderr),
                )
                duration_ms = (time.monotonic() - t0) * 1000
                await bus.publish(
                    CommandFinished(
                        command=raw,
                        exit_code=result.exit_code,
                        duration_ms=duration_ms,
                    )
                )
                ui.render(result)

            case InputKind.AGENT:
                query = agent_query(raw)
                context = await app.context_builder.build(app.shell)
                await bus.publish(
                    ContextCollected(
                        provider_count=app.context_builder.provider_count,
                        fragment_count=len(context),
                    )
                )
                app.state.conversation.append(
                    Message(role="user", content=query)
                )
                try:
                    final = await run_agent_loop(
                        agent=app.agent,
                        conversation=app.state.conversation,
                        context=context,
                        tools=app.tools,
                        permissions=app.permissions,
                        event_bus=bus,
                    )
                    ui.render(final)
                except AgentLoopLimitError as e:
                    print(f"[agentsh] {e}", file=sys.stderr)
                finally:
                    app.state.prune()
