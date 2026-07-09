"""Shared path canonicalization for file tools and permission keys."""

from collections.abc import Iterable
from pathlib import Path


def canonical_path(raw: str) -> Path:
    """Resolve raw to an absolute path with ~ and symlinks expanded.

    Permission evaluation and file IO must both use this so that
    alternate spellings of a path (./, ../, ~, symlinks) cannot bypass
    allow/deny rules that the actual read or write would honour.
    """
    return Path(raw).expanduser().resolve()


def is_within_roots(path: Path, roots: Iterable[Path]) -> bool:
    """Return True if path is one of roots, or nested inside one of them.

    Both path and every root must already be canonicalized (see
    canonical_path) so an alternate spelling of an out-of-bounds path
    (./, ../, ~, a symlink) cannot be mistaken for containment.
    """
    return any(path == root or root in path.parents for root in roots)
