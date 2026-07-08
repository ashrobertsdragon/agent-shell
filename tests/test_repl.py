"""Tests for the REPL main loop (run_repl) and the UI helper class."""

from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentsh.agent_loop import AgentLoopLimitError
from agentsh.app import App, AppState
from agentsh.classifier import InputKind
from agentsh.events import (
    CommandFinished,
    CommandStarted,
    ContextCollected,
    EventBus,
)
from agentsh.models import CommandResult, ContextFragment, Message
from agentsh.permissions import PermissionDeniedError, PermissionLevel
from agentsh.repl import UI, run_repl


@pytest.fixture(autouse=True)
def _isolated_home(monkeypatch: pytest.MonkeyPatch, tmp_path: object) -> None:
    """Redirect the REPL's on-disk history file away from the real home dir."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))


def _make_app(
    *,
    permission_level: PermissionLevel = PermissionLevel.ALLOW,
    context_fragments: list[ContextFragment] | None = None,
    provider_count: int = 0,
) -> tuple[App, AsyncMock]:
    """Build an App with fully mocked dependencies for REPL tests.

    Returns the App and the mocked RunCommand tool it wires in, since
    most SHELL-path assertions need to inspect that tool's calls.
    """
    shell = AsyncMock()
    shell.render_prompt = AsyncMock(return_value="$ ")
    shell.append_history = AsyncMock()

    run_command = AsyncMock()
    tools = MagicMock()
    tools.get.return_value = run_command

    permissions = MagicMock()
    permissions.evaluate.return_value = permission_level

    context_builder = MagicMock()
    context_builder.build = AsyncMock(return_value=context_fragments or [])
    context_builder.provider_count = provider_count

    app = App(
        shell=shell,
        tools=tools,
        permissions=permissions,
        context_builder=context_builder,
        agent=MagicMock(),
        state=AppState(),
        event_bus=EventBus(),
    )
    return app, run_command


async def _run_with_inputs(
    app: App, inputs: list[object], ui: MagicMock | None = None
) -> MagicMock:
    """Drive run_repl through a scripted sequence of prompt_async results.

    Each entry in inputs is either a string returned by prompt_async, or
    an exception instance raised by it (e.g. EOFError to end the loop).
    """
    session = MagicMock()
    session.prompt_async = AsyncMock(side_effect=inputs)
    ui = ui if ui is not None else MagicMock()
    if not isinstance(ui.confirm, AsyncMock):
        ui.confirm = AsyncMock(return_value=True)

    with (
        patch("agentsh.repl.PromptSession", return_value=session),
        patch("agentsh.repl.FileHistory"),
        patch("agentsh.repl.UI", return_value=ui),
    ):
        await run_repl(app)
    return ui


async def test_eof_breaks_loop() -> None:
    """run_repl returns cleanly when prompt_async raises EOFError."""
    app, _ = _make_app()
    shell = cast(AsyncMock, app.shell)
    await _run_with_inputs(app, [EOFError()])
    shell.render_prompt.assert_called_once()


async def test_keyboard_interrupt_continues_loop() -> None:
    """A KeyboardInterrupt during prompting is swallowed and the loop retries."""
    app, _ = _make_app()
    shell = cast(AsyncMock, app.shell)
    await _run_with_inputs(app, [KeyboardInterrupt(), EOFError()])
    assert shell.render_prompt.call_count == 2


async def test_blank_input_is_skipped() -> None:
    """Whitespace-only input is skipped without classification or history."""
    app, _ = _make_app()
    shell = cast(AsyncMock, app.shell)
    await _run_with_inputs(app, ["   ", EOFError()])
    shell.append_history.assert_not_called()
    assert shell.render_prompt.call_count == 2


async def test_shell_deny_skips_execution(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A DENY-level command is never sent to the RunCommand tool."""
    app, run_command = _make_app(permission_level=PermissionLevel.DENY)

    with patch(
        "agentsh.classifier.classify",
        new=AsyncMock(return_value=InputKind.SHELL),
    ):
        await _run_with_inputs(app, ["rm -rf /", EOFError()])

    run_command.invoke.assert_not_called()
    assert "denied" in capsys.readouterr().err


async def test_shell_confirm_declined_still_invokes_and_pairs_events(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A declined CONFIRM prompt is enforced inside RunCommand.invoke()
    itself (not pre-checked by the REPL, which only fast-paths DENY), so
    invoke() is still called; it raises PermissionDeniedError, which the
    REPL reports and pairs with a CommandFinished(exit_code=126) so the
    dangling CommandStarted from the DENY-only fast-path is never left
    unmatched.
    """
    app, run_command = _make_app(permission_level=PermissionLevel.CONFIRM)
    run_command.invoke = AsyncMock(
        side_effect=PermissionDeniedError("RunCommand denied by user: x")
    )
    finished: list[CommandFinished] = []
    app.event_bus.subscribe(CommandFinished, finished.append)

    with patch(
        "agentsh.classifier.classify",
        new=AsyncMock(return_value=InputKind.SHELL),
    ):
        await _run_with_inputs(app, ["git commit -m x", EOFError()])

    run_command.invoke.assert_called_once_with(command="git commit -m x")
    assert "denied by user" in capsys.readouterr().err
    assert [e.exit_code for e in finished] == [126]


async def test_shell_success_publishes_events_and_renders() -> None:
    """A successful command publishes start/finish events and renders output."""
    app, run_command = _make_app()
    run_command.invoke = AsyncMock(
        return_value=CommandResult(
            stdout="hi\n", stderr="", exit_code=0, duration_ms=1.0, cwd="/tmp"
        )
    )
    started: list[CommandStarted] = []
    finished: list[CommandFinished] = []
    app.event_bus.subscribe(CommandStarted, started.append)
    app.event_bus.subscribe(CommandFinished, finished.append)
    ui = MagicMock()

    with patch(
        "agentsh.classifier.classify",
        new=AsyncMock(return_value=InputKind.SHELL),
    ):
        await _run_with_inputs(app, ["echo hi", EOFError()], ui=ui)

    run_command.invoke.assert_called_once_with(command="echo hi")
    assert [e.command for e in started] == ["echo hi"]
    assert [e.exit_code for e in finished] == [0]
    ui.render.assert_called_once()


async def test_shell_permission_denied_error_is_handled(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A PermissionDeniedError raised by invoke() is caught and reported."""
    app, run_command = _make_app()
    run_command.invoke = AsyncMock(
        side_effect=PermissionDeniedError("blocked by policy")
    )

    with patch(
        "agentsh.classifier.classify",
        new=AsyncMock(return_value=InputKind.SHELL),
    ):
        await _run_with_inputs(app, ["rm -rf /", EOFError()])

    assert "blocked by policy" in capsys.readouterr().err


async def test_agent_path_builds_context_and_runs_loop() -> None:
    """AGENT input builds context, runs the agent loop, and renders the result."""
    fragment = ContextFragment(provider="git", summary="s", payload={})
    app, _ = _make_app(context_fragments=[fragment], provider_count=3)
    collected: list[ContextCollected] = []
    app.event_bus.subscribe(ContextCollected, collected.append)
    final = Message(role="assistant", content="done")
    ui = MagicMock()

    with (
        patch(
            "agentsh.classifier.classify",
            new=AsyncMock(return_value=InputKind.AGENT),
        ),
        patch(
            "agentsh.agent_loop.run_agent_loop",
            new=AsyncMock(return_value=final),
        ) as mock_loop,
    ):
        await _run_with_inputs(app, ["/agent do a thing", EOFError()], ui=ui)

    mock_loop.assert_called_once()
    assert app.state.conversation[0].role == "user"
    assert app.state.conversation[0].content == "do a thing"
    ui.render.assert_called_once_with(final)
    assert [(c.provider_count, c.fragment_count) for c in collected] == [(3, 1)]


async def test_agent_loop_limit_error_is_handled(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """AgentLoopLimitError is caught, reported, and state is still pruned."""
    app, _ = _make_app()

    with (
        patch(
            "agentsh.classifier.classify",
            new=AsyncMock(return_value=InputKind.AGENT),
        ),
        patch(
            "agentsh.agent_loop.run_agent_loop",
            new=AsyncMock(
                side_effect=AgentLoopLimitError("exceeded 20 iterations")
            ),
        ),
    ):
        await _run_with_inputs(app, ["/agent loop forever", EOFError()])

    assert "exceeded 20 iterations" in capsys.readouterr().err
    assert len(app.state.conversation) == 1


class TestUI:
    """Direct unit tests for the UI helper class, independent of run_repl."""

    def test_render_command_result_prints_stdout_and_stderr(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """stdout goes to stdout and stderr goes to stderr."""
        ui = UI(MagicMock())
        ui.render(
            CommandResult(
                stdout="out\n",
                stderr="err\n",
                exit_code=1,
                duration_ms=1.0,
                cwd="/tmp",
            )
        )
        captured = capsys.readouterr()
        assert captured.out == "out\n"
        assert captured.err == "err\n"

    def test_render_message_prints_content_when_present(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A Message with content is printed to stdout."""
        ui = UI(MagicMock())
        ui.render(Message(role="assistant", content="hello"))
        assert capsys.readouterr().out == "hello\n"

    def test_render_message_prints_nothing_when_empty(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A Message with empty content produces no output."""
        ui = UI(MagicMock())
        ui.render(Message(role="assistant", content=""))
        assert capsys.readouterr().out == ""

    async def test_confirm_yes_returns_true(self) -> None:
        """Typing 'y' confirms the prompt."""
        session = MagicMock()
        session.prompt_async = AsyncMock(return_value="y")
        assert await UI(session).confirm("RunCommand", {"command": "ls"})

    async def test_confirm_anything_else_returns_false(self) -> None:
        """Any answer other than 'y' denies the prompt."""
        session = MagicMock()
        session.prompt_async = AsyncMock(return_value="n")
        assert not await UI(session).confirm("RunCommand", {"command": "ls"})

    async def test_confirm_eof_denies(self) -> None:
        """EOF while confirming is treated as a denial, not a crash."""
        session = MagicMock()
        session.prompt_async = AsyncMock(side_effect=EOFError())
        assert not await UI(session).confirm("RunCommand", {"command": "ls"})

    async def test_confirm_keyboard_interrupt_denies(self) -> None:
        """Ctrl-C while confirming is treated as a denial, not a crash."""
        session = MagicMock()
        session.prompt_async = AsyncMock(side_effect=KeyboardInterrupt())
        assert not await UI(session).confirm("RunCommand", {"command": "ls"})
