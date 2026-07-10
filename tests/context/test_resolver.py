"""Tests for dynamic context provider resolution."""

import sys
import types

import pytest

from agentsh.config import ContextConfig
from agentsh.context import providers as providers_module
from agentsh.context.providers import (
    UnknownProviderError,
    build_providers,
    register,
    resolve_provider,
)
from agentsh.models import ContextFragment
from agentsh.shell.protocol import Shell


def test_resolve_provider_returns_factory_by_module_name() -> None:
    """A provider name maps to the class in its module."""
    factory = resolve_provider("git")
    assert type(factory()).__name__ == "GitProvider"


def test_resolve_provider_handles_all_defaults() -> None:
    """Every default-configured provider resolves."""
    for name in ContextConfig().providers:
        resolve_provider(name)


def test_resolve_provider_unknown_name_raises() -> None:
    """A bad config value produces a clear error, not an ImportError."""
    with pytest.raises(UnknownProviderError, match="nonexistent"):
        resolve_provider("nonexistent")


def test_build_providers_preserves_config_order() -> None:
    """Providers are instantiated in the configured order."""
    built = build_providers(["python", "git"])
    assert [type(p).__name__ for p in built] == [
        "PythonProvider",
        "GitProvider",
    ]


def test_default_config_includes_history_and_environment() -> None:
    """History and environment context are on by default."""
    providers = ContextConfig().providers
    assert "history" in providers
    assert "environment" in providers


def test_resolve_provider_does_not_guess_class_name_from_title_case(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resolution relies on @register, not on `name.title() + "Provider"`.

    This is the bug issue #23 describes: the old resolver did
    `getattr(module, f"{name.title()}Provider")`, which mangles
    multi-word names ("node_env".title() -> "Node_env", not "NodeEnv").
    A fake provider module with a deliberately unconventional class
    name proves resolution no longer depends on that convention.
    """
    module_name = "agentsh.context.providers.node_env"
    fake_module = types.ModuleType(module_name)

    @register("node_env")
    class TotallyUnconventionalName:
        name = "node_env"

        async def collect(self, shell: Shell) -> ContextFragment | None:
            """Satisfy the ContextProvider protocol; unused by this test."""
            return None

    setattr(fake_module, "TotallyUnconventionalName", TotallyUnconventionalName)
    monkeypatch.setitem(sys.modules, module_name, fake_module)

    try:
        factory = resolve_provider("node_env")
        assert factory is TotallyUnconventionalName
    finally:
        providers_module._registry._entries.pop("node_env", None)


def test_resolve_provider_module_not_found_raises_unknown_provider_error() -> (
    None
):
    """A provider name with no backing module is a clear UnknownProviderError."""
    with pytest.raises(UnknownProviderError, match="never_created"):
        resolve_provider("never_created")
