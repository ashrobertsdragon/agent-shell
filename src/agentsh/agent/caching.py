"""Per-instance identity-keyed memoization for expensive request-building.

Backends convert `context` (environment fragments) and `tools` (JSON
schemas) into provider-specific request payloads. Within a single user
turn, `run_agent_loop` calls `Agent.respond` once per iteration (up to
`max_iterations` times), and `context` plus the tool schema list are the
same object every iteration -- only the growing `conversation` differs.
Rebuilding the system prompt string and provider tool list from scratch
on every iteration is therefore pure waste for the common case.

`IdentityCache` memoizes by object identity (`is`, not `==`) rather than
value equality: the cache key is the exact object passed in, so a new
turn (which builds a fresh `context`/`tools` list) transparently
invalidates the cache with no explicit invalidation logic required. A
list mutated in place rather than replaced is *not* detected as a
change -- callers must treat the key object as immutable for the
duration of a turn, which matches how `run_agent_loop` and the context
builder already use it.
"""

from collections.abc import Callable
from typing import cast

_MISSING = object()


class IdentityCache[T]:
    """Single-slot cache keyed on the identity of the last-seen key."""

    def __init__(self) -> None:
        """Start with an empty cache slot."""
        self._key: object = _MISSING
        self._value: object = _MISSING

    def get_or_build(self, key: object, build: Callable[[], T]) -> T:
        """Return the value for `key`, building and caching it on a miss.

        A miss is either an empty slot or a `key` that is not (by
        identity) the object cached last time.
        """
        if self._key is key and self._value is not _MISSING:
            return cast(T, self._value)
        value = build()
        self._key = key
        self._value = value
        return value
