"""Tests for the CLI entry point in agentsh.main."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentsh.config import (
    AgentConfig,
    Config,
    ContextConfig,
    PermissionRulesConfig,
)
from agentsh.context.providers import UnknownProviderError
from agentsh.main import _build_app, main
from agentsh.shell import UnsupportedShellError


def _config() -> Config:
    """Build a representative Config for wiring tests."""
    return Config(
        shell="bash",
        agent=AgentConfig(provider="anthropic"),
        context=ContextConfig(providers=["git"], timeout_ms=150),
        permissions=PermissionRulesConfig(allow={"ReadFile:*"}),
    )


def test_build_app_wires_dependencies() -> None:
    """_build_app wires the config-derived shell, tools, and agent into App."""
    config = _config()
    fake_shell = MagicMock()
    fake_providers = [MagicMock()]
    fake_agent_instance = MagicMock()
    fake_agent_cls = MagicMock(return_value=fake_agent_instance)

    with (
        patch("agentsh.main.load_config", return_value=config),
        patch(
            "agentsh.main.create_shell", return_value=fake_shell
        ) as mock_create_shell,
        patch(
            "agentsh.main.build_providers", return_value=fake_providers
        ) as mock_build_providers,
        patch("agentsh.main.Agent") as mock_agent,
    ):
        mock_agent.from_provider.return_value = fake_agent_cls
        app = _build_app()

    mock_create_shell.assert_called_once_with("bash")
    mock_build_providers.assert_called_once_with(["git"])
    mock_agent.from_provider.assert_called_once_with("anthropic")
    fake_agent_cls.assert_called_once_with(config.agent)

    assert app.shell is fake_shell
    assert app.agent is fake_agent_instance
    assert app.context_builder.provider_count == len(fake_providers)
    assert {
        app.tools.get("RunCommand").name,
        app.tools.get("ReadFile").name,
        app.tools.get("WriteFile").name,
    } == {"RunCommand", "ReadFile", "WriteFile"}


def test_main_success_runs_repl() -> None:
    """main builds the app then runs the REPL loop to completion."""
    fake_app = MagicMock()

    with (
        patch("agentsh.main._build_app", return_value=fake_app),
        patch("agentsh.main.run_repl", new=AsyncMock()) as mock_run_repl,
    ):
        main()

    mock_run_repl.assert_awaited_once_with(fake_app)


def test_main_exits_on_unsupported_shell_error() -> None:
    """An UnsupportedShellError from _build_app exits instead of raising."""
    with (
        patch(
            "agentsh.main._build_app",
            side_effect=UnsupportedShellError("no shell backend for 'fish'"),
        ),
        patch("agentsh.main.run_repl", new=AsyncMock()) as mock_run_repl,
        pytest.raises(SystemExit) as exc_info,
    ):
        main()

    assert "agentsh:" in str(exc_info.value)
    assert "no shell backend for 'fish'" in str(exc_info.value)
    mock_run_repl.assert_not_awaited()


def test_main_exits_on_unknown_provider_error() -> None:
    """An UnknownProviderError from _build_app exits instead of raising."""
    with (
        patch(
            "agentsh.main._build_app",
            side_effect=UnknownProviderError("Unknown context provider: 'foo'"),
        ),
        patch("agentsh.main.run_repl", new=AsyncMock()) as mock_run_repl,
        pytest.raises(SystemExit) as exc_info,
    ):
        main()

    assert "agentsh:" in str(exc_info.value)
    assert "Unknown context provider: 'foo'" in str(exc_info.value)
    mock_run_repl.assert_not_awaited()
