"""Tests for IdentityCache, the per-instance memoization helper backends
use to avoid rebuilding the system prompt / tool schema on every
agent-loop iteration within a single user turn.
"""

from agentsh.agent.caching import IdentityCache


def test_first_call_invokes_builder() -> None:
    """An empty cache always builds on the first call."""
    cache: IdentityCache[int] = IdentityCache()

    result = cache.get_or_build(object(), lambda: 42)

    assert result == 42


def test_same_key_object_reuses_cached_value() -> None:
    """A second call with the identical key object skips the builder."""
    cache: IdentityCache[str] = IdentityCache()
    calls: list[int] = []

    def build() -> str:
        calls.append(1)
        return "built"

    key: list[str] = ["a"]
    first = cache.get_or_build(key, build)
    second = cache.get_or_build(key, build)

    assert first == "built"
    assert second == "built"
    assert len(calls) == 1


def test_different_key_object_rebuilds_even_if_equal_by_value() -> None:
    """Identity, not equality, decides the cache hit -- a new list with
    the same contents is treated as a new turn and forces a rebuild.
    """
    cache: IdentityCache[str] = IdentityCache()
    calls: list[int] = []

    def build() -> str:
        calls.append(1)
        return f"built-{len(calls)}"

    first = cache.get_or_build(["a"], build)
    second = cache.get_or_build(["a"], build)

    assert first == "built-1"
    assert second == "built-2"
    assert len(calls) == 2


def test_falsy_cached_value_is_still_returned_from_cache() -> None:
    """A falsy but valid cached value (empty list, 0, "") must not be
    mistaken for an empty cache slot on the next lookup.
    """
    cache: IdentityCache[list[str]] = IdentityCache()
    calls: list[int] = []

    def build() -> list[str]:
        calls.append(1)
        return []

    key = object()
    first = cache.get_or_build(key, build)
    second = cache.get_or_build(key, build)

    assert first == []
    assert second == []
    assert len(calls) == 1
