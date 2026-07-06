"""Shared path canonicalization for file tools and permission keys."""

from pathlib import Path


def canonical_path(raw: str) -> Path:
    """Resolve raw to an absolute path with ~ and symlinks expanded.

    Permission evaluation and file IO must both use this so that
    alternate spellings of a path (./, ../, ~, symlinks) cannot bypass
    allow/deny rules that the actual read or write would honour.
    """
    return Path(raw).expanduser().resolve()
