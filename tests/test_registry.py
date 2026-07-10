"""Tests for the generic decorator-based plugin registry (issue #23).

This registry is the single primitive shared by shell backends, context
providers, and agent backends -- see agentsh.registry, agentsh.shell.
_registry, agentsh.context.providers, and agentsh.agent.base.
"""

from pathlib import Path

import pytest

from agentsh.registry import Registry, discover_modules


class _Plugin:
    """Minimal stand-in for a registrable plugin class."""


def test_register_then_get_returns_the_registered_class() -> None:
    """A class registered under a name is returned by get(name)."""
    registry: Registry[_Plugin] = Registry()

    @registry.register("widget")
    class Widget(_Plugin):
        pass

    assert registry.get("widget") is Widget


def test_get_is_case_insensitive() -> None:
    """Lookup normalizes case, matching how names arrive from config."""
    registry: Registry[_Plugin] = Registry()

    @registry.register("Widget")
    class Widget(_Plugin):
        pass

    assert registry.get("widget") is Widget
    assert registry.get("WIDGET") is Widget


def test_get_unknown_name_raises_key_error() -> None:
    """An unregistered name is a clear KeyError, not a silent None."""
    registry: Registry[_Plugin] = Registry()
    with pytest.raises(KeyError):
        registry.get("nonexistent")


def test_available_lists_registered_names_sorted() -> None:
    """available() reflects every registration, alphabetically."""
    registry: Registry[_Plugin] = Registry()
    registry.register("zeta")(type("Zeta", (_Plugin,), {}))
    registry.register("alpha")(type("Alpha", (_Plugin,), {}))
    assert registry.available() == ["alpha", "zeta"]


def test_registration_is_independent_of_the_class_name() -> None:
    """The registered name need not match the class name in any way.

    This is the core fix for issue #23: the old context-provider and
    agent-backend resolvers guessed a class name from the provider name
    via `str.title()`, which silently broke on multi-word or
    already-capitalized names (e.g. "node_env" -> "Node_envProvider",
    "openrouter" -> "OpenrouterAgent" instead of "OpenRouterAgent").
    A decorator-registered class carries no such constraint.
    """
    registry: Registry[_Plugin] = Registry()

    @registry.register("node_env")
    class TotallyUnrelatedName(_Plugin):
        pass

    assert registry.get("node_env") is TotallyUnrelatedName


def test_discover_modules_imports_every_non_underscore_module(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """discover_modules imports plain modules and skips underscore-prefixed ones."""
    import sys

    package_dir = tmp_path / "fake_plugins"
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "real_plugin.py").write_text(
        "registered = True\n", encoding="utf-8"
    )
    (package_dir / "_private.py").write_text(
        "raise RuntimeError('must not be imported')\n", encoding="utf-8"
    )

    monkeypatch.syspath_prepend(str(tmp_path))
    for name in list(sys.modules):
        if name.startswith("fake_plugins"):
            del sys.modules[name]

    discover_modules(package_dir, "fake_plugins")

    assert "fake_plugins.real_plugin" in sys.modules
    assert "fake_plugins._private" not in sys.modules
