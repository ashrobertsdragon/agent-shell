"""Tests for config loading."""

import textwrap
from pathlib import Path

from agentsh.config import Config, load_config


def test_load_defaults_when_no_file(tmp_path: Path) -> None:
    """Missing config file returns defaults."""
    cfg = load_config(tmp_path / "nonexistent.toml")
    assert cfg.agent.provider == "anthropic"
    assert cfg.shell == "auto"


def test_load_overrides_from_file(tmp_path: Path) -> None:
    """Values in the TOML file override defaults."""
    config_file = tmp_path / "config.toml"
    config_file.write_text(
        textwrap.dedent("""
        shell = "bash"

        [agent]
        provider = "anthropic"
        model = "claude-haiku-4-5-20251001"
        web_fetch = false
    """)
    )
    cfg = load_config(config_file)
    assert cfg.shell == "bash"
    assert cfg.agent.provider == "anthropic"
    assert cfg.agent.model == "claude-haiku-4-5-20251001"
    assert cfg.agent.web_fetch is False


def test_permission_rules_default_empty(tmp_path: Path) -> None:
    """Default config has no permission rules."""
    cfg = load_config(tmp_path / "no.toml")
    assert cfg.permissions.allow == set()
    assert cfg.permissions.confirm == set()
    assert cfg.permissions.deny == set()


def test_load_config_returns_config_type(tmp_path: Path) -> None:
    """load_config always returns a Config instance."""
    assert isinstance(load_config(tmp_path / "missing.toml"), Config)
