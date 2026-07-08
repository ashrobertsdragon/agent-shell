"""Tests for EventBus publish/subscribe ordering and subscriber isolation."""

import asyncio
import logging
import time

import pytest

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


async def test_subscriber_exception_is_logged_not_silently_swallowed(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A failing handler's exception is logged with a traceback, not dropped."""
    bus = EventBus()

    def bad(e: object) -> None:
        raise RuntimeError("boom")

    bus.subscribe(CommandFinished, bad)
    with caplog.at_level(logging.ERROR, logger="agentsh.events"):
        await bus.publish(
            CommandFinished(command="echo", exit_code=0, duration_ms=0.1)
        )

    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert "Unhandled exception in event subscriber" in record.message
    assert record.exc_info is not None
    assert "RuntimeError: boom" in caplog.text


async def test_slow_subscriber_is_not_currently_bounded_by_a_timeout() -> None:
    """KNOWN GAP: publish() has no per-subscriber timeout.

    EventBus.publish dispatches subscribers synchronously inline; a
    slow subscriber blocks the whole publish() call — and, in the REPL
    hot path, the REPL itself — for as long as it takes. This test
    pins that current behavior as a documented gap rather than a silent
    assumption: adding a bounded per-subscriber timeout (e.g. via
    asyncio.wait_for) would require this test to be rewritten to assert
    the opposite (that publish() returns in roughly the timeout, not
    the subscriber's full delay).
    """
    bus = EventBus()
    delay = 0.05

    async def slow(e: object) -> None:
        await asyncio.sleep(delay)

    bus.subscribe(CommandFinished, slow)
    start = time.monotonic()
    await bus.publish(
        CommandFinished(command="sleep", exit_code=0, duration_ms=0.1)
    )
    elapsed = time.monotonic() - start
    assert elapsed >= delay
