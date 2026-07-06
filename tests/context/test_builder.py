"""Tests for ContextBuilder timeout and failure isolation."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from agentsh.context.builder import ContextBuilder
from agentsh.models import ContextFragment


@pytest.fixture
def shell() -> MagicMock:
    """Minimal shell mock."""
    return MagicMock()


async def test_builder_collects_fragments(shell: MagicMock) -> None:
    """build returns fragments from all successful providers."""
    frag = ContextFragment(provider="test", summary="test", payload={})
    provider = MagicMock()
    provider.collect = AsyncMock(return_value=frag)
    builder = ContextBuilder(providers=[provider], timeout_ms=200)
    result = await builder.build(shell)
    assert result == [frag]


async def test_builder_swallows_failures(shell: MagicMock) -> None:
    """build drops providers that raise exceptions."""
    provider = MagicMock()
    provider.collect = AsyncMock(side_effect=RuntimeError("boom"))
    builder = ContextBuilder(providers=[provider], timeout_ms=200)
    result = await builder.build(shell)
    assert result == []


async def test_builder_times_out_slow_provider(shell: MagicMock) -> None:
    """build drops providers that exceed the timeout."""

    async def slow(_: object) -> ContextFragment:
        await asyncio.sleep(10)
        return ContextFragment(provider="slow", summary="slow", payload={})

    provider = MagicMock()
    provider.collect = slow
    builder = ContextBuilder(providers=[provider], timeout_ms=50)
    result = await builder.build(shell)
    assert result == []
