"""Tests for EventBus publish/subscribe ordering and subscriber isolation."""

import asyncio

from agentsh.events import CommandFinished, EventBus, ToolDenied


async def test_subscriber_receives_event() -> None:
    """A subscribed handler receives the published event."""
    bus = EventBus()
    received: list[CommandFinished] = []
    bus.subscribe(CommandFinished, received.append)
    event = CommandFinished(command="ls", exit_code=0, duration_ms=1.0)
    await bus.publish(event)
    assert received == [event]


async def test_multiple_subscribers_all_called() -> None:
    """All handlers subscribed to a type are called in registration order."""
    bus = EventBus()
    log: list[str] = []
    bus.subscribe(CommandFinished, lambda e: log.append("first"))
    bus.subscribe(CommandFinished, lambda e: log.append("second"))
    await bus.publish(
        CommandFinished(command="pwd", exit_code=0, duration_ms=0.5)
    )
    assert log == ["first", "second"]


async def test_subscriber_exception_does_not_stop_others() -> None:
    """A failing handler does not prevent subsequent handlers from running."""
    bus = EventBus()
    log: list[str] = []

    async def bad(e: object) -> None:
        raise RuntimeError("boom")

    bus.subscribe(CommandFinished, bad)
    bus.subscribe(CommandFinished, lambda e: log.append("ok"))
    await bus.publish(
        CommandFinished(command="echo", exit_code=0, duration_ms=0.1)
    )
    assert log == ["ok"]


def test_unrelated_event_not_delivered() -> None:
    """Handlers are only called for their subscribed event type."""
    bus = EventBus()
    received: list[object] = []
    bus.subscribe(CommandFinished, received.append)
    asyncio.run(
        bus.publish(
            ToolDenied(tool_name="RunCommand", key="RunCommand:rm -rf /")
        )
    )
    assert received == []
