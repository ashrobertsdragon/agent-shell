"""Generic decorator-based plugin registry.

Shared by shell backends, context providers, and agent backends
(issue #23) so all three extension points resolve plugins the same
way -- an explicit `@register(name)` decorator maps a name to a class
-- instead of each inventing its own mechanism. Shell backends are
discovered eagerly via `discover_modules` (every module in the plugin
directory is imported up front, since none of them carry optional
third-party dependencies); context providers and agent backends are
resolved lazily by importing only the one module named by a config
value, since agent backends in particular depend on optional
per-provider SDKs that may not be installed.
"""

from collections.abc import Callable
from importlib import import_module
from pathlib import Path


class Registry[T]:
    """Maps lowercase names to classes registered via a decorator."""

    def __init__(self) -> None:
        """Initialise an empty registry."""
        self._entries: dict[str, type[T]] = {}

    def register(self, name: str) -> Callable[[type[T]], type[T]]:
        """Return a decorator that registers the decorated class under name.

        Raises:
            ValueError: If a different class is already registered under
                this name (re-registering the same class, e.g. from a
                module re-imported in tests, is a no-op).
        """

        def _decorator(cls: type[T]) -> type[T]:
            key = name.lower()
            existing = self._entries.get(key)
            if existing is not None and existing is not cls:
                raise ValueError(
                    f"{key!r} is already registered to "
                    f"{existing.__qualname__}; cannot also register "
                    f"{cls.__qualname__}"
                )
            self._entries[key] = cls
            return cls

        return _decorator

    def get(self, name: str) -> type[T]:
        """Return the class registered under name.

        Raises:
            KeyError: If no class is registered under name.
        """
        return self._entries[name.lower()]

    def available(self) -> list[str]:
        """Return every registered name, sorted."""
        return sorted(self._entries.keys())


def discover_modules(package_dir: Path, package: str) -> None:
    """Import every non-underscore-prefixed module in package_dir.

    Importing a module runs any `@registry.register(name)` decorator in
    it as a side effect, which is how eager, complete registration is
    achieved for a plugin set with no optional dependencies.
    """
    for module in package_dir.glob("*.py"):
        if module.name.startswith("_"):
            continue
        import_module(f".{module.stem}", package=package)
