"""Tests for shell plugin discovery."""

from pathlib import Path

import agentsh.shell.plugin
from agentsh.shell import _registry
from agentsh.shell.plugin import register_plugins


def test_register_plugins_registers_builtin_shells() -> None:
    """All built-in shell backends are discoverable after registration."""
    register_plugins()
    assert {"bash", "cmd", "fish", "nu", "powershell", "zsh"} <= set(
        _registry.available()
    )


def test_register_plugins_ignores_non_module_entries() -> None:
    """Stray files and directories in the plugin dir do not break discovery."""
    plugin_dir = Path(str(agentsh.shell.plugin.__file__)).parent
    stray = plugin_dir / "stray_data.txt"
    stray.write_text("not a module", encoding="utf-8")
    try:
        register_plugins()
    finally:
        stray.unlink()
