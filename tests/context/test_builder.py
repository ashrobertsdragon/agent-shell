"""Tests for ContextBuilder timeout and failure isolation."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from agentsh.context.builder import ContextBuilder
from agentsh.models import ContextFragment
from agentsh.shell.plugin.bash import BashShell


@pytest.fixture
def shell() -> AsyncMock:
    """Minimal shell mock with async methods (e.g. `reset`)."""
    return AsyncMock()


async def test_builder_timeout_restarts_real_shell_subprocess() -> None:
    """A timeout during collection kills and restarts a real subprocess.

    End-to-end regression test for the timeout-corruption bug: without
    the fix, the subprocess left behind by the abandoned `sleep`
    command would still be alive, and its eventual sentinel line would
    bleed into the next `execute` call's output.
    """
    real_shell = BashShell()
    try:
        proc_before = await real_shell.process
        pid_before = proc_before.pid

        async def slow(_: object) -> ContextFragment | None:
            await real_shell.execute("sleep 5")
            return None

        provider = MagicMock()
        provider.collect = slow
        builder = ContextBuilder(providers=[provider], timeout_ms=50)
        await builder.build(real_shell)

        assert proc_before.returncode is not None

        proc_after = await real_shell.process
        assert proc_after.pid != pid_before

        result = await real_shell.execute("echo fresh")
        assert result.stdout.strip() == "fresh"
        assert result.exit_code == 0
    finally:
        await real_shell.close()


async def test_builder_collects_fragments(shell: AsyncMock) -> None:
    """build returns fragments from all successful providers."""
    frag = ContextFragment(provider="test", summary="test", payload={})
    provider = MagicMock()
    provider.collect = AsyncMock(return_value=frag)
    builder = ContextBuilder(providers=[provider], timeout_ms=200)
    result = await builder.build(shell)
    assert result == [frag]
    shell.reset.assert_not_called()


async def test_builder_swallows_failures(shell: AsyncMock) -> None:
    """build drops providers that raise exceptions."""
    provider = MagicMock()
    provider.collect = AsyncMock(side_effect=RuntimeError("boom"))
    builder = ContextBuilder(providers=[provider], timeout_ms=200)
    result = await builder.build(shell)
    assert result == []
    shell.reset.assert_not_called()


async def test_builder_times_out_slow_provider(shell: AsyncMock) -> None:
    """build drops providers that exceed the timeout."""

    async def slow(_: object) -> ContextFragment:
        await asyncio.sleep(10)
        return ContextFragment(provider="slow", summary="slow", payload={})

    provider = MagicMock()
    provider.collect = slow
    builder = ContextBuilder(providers=[provider], timeout_ms=50)
    result = await builder.build(shell)
    assert result == []


async def test_builder_resets_shell_on_timeout(shell: AsyncMock) -> None:
    """A provider timeout triggers a shell reset.

    This ensures the abandoned command's subprocess is killed rather
    than left running to corrupt a later `execute` call.
    """

    async def slow(_: object) -> ContextFragment:
        await asyncio.sleep(10)
        return ContextFragment(provider="slow", summary="slow", payload={})

    provider = MagicMock()
    provider.collect = slow
    builder = ContextBuilder(providers=[provider], timeout_ms=50)
    await builder.build(shell)
    shell.reset.assert_awaited_once()
