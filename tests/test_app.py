"""Tests for AppState conversation pruning."""

from agentsh.app import AppState
from agentsh.models import Message


def _turn(query: str, tool_messages: int) -> list[Message]:
    """Build one agent turn: a user message followed by tool traffic."""
    turn = [Message(role="user", content=query)]
    for i in range(tool_messages):
        turn.append(Message(role="assistant", content=f"call {i}"))
        turn.append(Message(role="tool", content=f"result {i}"))
    turn.append(Message(role="assistant", content="done"))
    return turn


def test_prune_noop_when_under_limit() -> None:
    """Conversations within max_history are untouched."""
    state = AppState(conversation=_turn("hi", 1), max_history=10)
    before = list(state.conversation)
    state.prune()
    assert state.conversation == before


def test_prune_cuts_at_oldest_user_message_within_limit() -> None:
    """Pruning drops whole turns and always starts on a user message."""
    state = AppState(
        conversation=_turn("one", 2) + _turn("two", 1) + _turn("three", 1),
        max_history=10,
    )
    state.prune()
    assert len(state.conversation) <= 10
    assert state.conversation[0].role == "user"
    assert state.conversation[0].content == "two"


def test_prune_never_splits_most_recent_turn() -> None:
    """A turn longer than max_history is kept whole, not sliced mid-turn."""
    state = AppState(
        conversation=_turn("old", 1) + _turn("big", 8),
        max_history=10,
    )
    state.prune()
    assert state.conversation[0].role == "user"
    assert state.conversation[0].content == "big"
    assert len(state.conversation) == 18


def test_prune_without_user_messages_leaves_conversation_intact() -> None:
    """No user message means no safe cut point; nothing is dropped."""
    conversation = [
        Message(role="assistant", content=str(i)) for i in range(15)
    ]
    state = AppState(conversation=list(conversation), max_history=10)
    state.prune()
    assert state.conversation == conversation
