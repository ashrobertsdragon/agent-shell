"""Tests for Agent.from_provider's decorator-based backend resolution.

Issue #23: the old resolver did
`getattr(module, f"{provider.title()}Agent")`, silently breaking on any
backend module whose class name didn't match that convention (and
raising an unhelpful AttributeError when it did). `from_provider` now
imports the requested backend module -- to trigger its
`@register(name)` decorator -- and looks the class up in a registry,
never inspecting the class's name.
"""

import sys
import types

import pytest

from agentsh.agent.base import Agent, register


def test_from_provider_does_not_guess_class_name_from_title_case(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resolution relies on @register, not on `provider.title() + "Agent"`.

    A fake backend module with a deliberately unconventional class name
    proves resolution no longer depends on that convention.
    """
    module_name = "agentsh.agent.made_up_provider"
    fake_module = types.ModuleType(module_name)

    @register("made_up_provider")
    class TotallyUnconventionalName(Agent):
        pass

    setattr(fake_module, "TotallyUnconventionalName", TotallyUnconventionalName)
    monkeypatch.setitem(sys.modules, module_name, fake_module)

    resolved = Agent.from_provider("made_up_provider")
    assert resolved is TotallyUnconventionalName


def test_from_provider_is_case_insensitive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A provider name resolves regardless of the case used to register it."""
    module_name = "agentsh.agent.mixed_case_provider"
    fake_module = types.ModuleType(module_name)

    @register("Mixed_Case_Provider")
    class MixedCaseAgent(Agent):
        pass

    setattr(fake_module, "MixedCaseAgent", MixedCaseAgent)
    monkeypatch.setitem(sys.modules, module_name, fake_module)

    assert Agent.from_provider("mixed_case_provider") is MixedCaseAgent
    assert Agent.from_provider("MIXED_CASE_PROVIDER") is MixedCaseAgent


def test_from_provider_unknown_module_raises_module_not_found_error() -> None:
    """A provider name with no backing module fails clearly, not silently."""
    with pytest.raises(ModuleNotFoundError):
        Agent.from_provider("never_created_provider")


def test_from_provider_resolves_real_anthropic_backend() -> None:
    """End-to-end sanity check against the real anthropic.py module."""
    from agentsh.agent.anthropic import AnthropicAgent

    assert Agent.from_provider("anthropic") is AnthropicAgent
