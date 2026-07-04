"""Tests for config loading."""

import textwrap
from pathlib import Path

from agentsh.config import Config, load_config


def test_load_defaults_when_no_file(tmp_path: Path) -> None:
    """Missing config file returns defaults."""
    cfg = load_config(tmp_path / "nonexistent.toml")
    assert cfg.agent.default == "anthropic"
    assert cfg.shell.backend == "auto"


def test_load_overrides_from_file(tmp_path: Path) -> None:
    """Values in the TOML file override defaults."""
    config_file = tmp_path / "config.toml"
    config_file.write_text(
        textwrap.dedent("""
        [shell]
        backend = "bash"

        [agent]
        default = "openai"

        [agent.anthropic]
        model = "claude-haiku-4-5-20251001"
        web_fetch = false
    """)
    )
    cfg = load_config(config_file)
    assert cfg.shell.backend == "bash"
    assert cfg.agent.default == "openai"
    assert cfg.agent.backends["anthropic"].model == "claude-haiku-4-5-20251001"
    assert cfg.agent.backends["anthropic"].web_fetch is False


def test_permission_rules_default_empty(tmp_path: Path) -> None:
    """Default config has no permission rules."""
    cfg = load_config(tmp_path / "no.toml")
    assert cfg.permissions.allow == ()
    assert cfg.permissions.confirm == ()
    assert cfg.permissions.deny == ()


def test_load_config_returns_config_type(tmp_path: Path) -> None:
    """load_config always returns a Config instance."""
    assert isinstance(load_config(tmp_path / "missing.toml"), Config)
