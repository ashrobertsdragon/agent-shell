"""Tests for dynamic context provider resolution."""

import pytest

from agentsh.config import ContextConfig
from agentsh.context.providers import (
    UnknownProviderError,
    build_providers,
    resolve_provider,
)


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
