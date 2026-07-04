"""Tests for core data models."""

from agentsh.models import CommandResult, ContextFragment, Message, ToolResult


def test_command_result_is_frozen() -> None:
    r = CommandResult(stdout="hi", stderr="", exit_code=0, duration_ms=1.0, cwd="/")
    try:
        r.stdout = "x"  # type: ignore[misc]
        raise AssertionError("should be frozen")
    except Exception:
        pass


def test_message_defaults() -> None:
    m = Message(role="user", content="hello")
    assert m.tool_calls == ()
    assert m.tool_results == ()


def test_tool_result_default_not_error() -> None:
    tr = ToolResult(call_id="abc", content="ok")
    assert not tr.is_error


def test_context_fragment_roundtrip() -> None:
    cf = ContextFragment(
        provider="git", summary="branch main", payload={"branch": "main"}
    )
    assert cf.payload["branch"] == "main"
