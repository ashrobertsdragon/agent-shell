"""Config dataclasses and TOML loader for ~/.config/agentsh/config.toml."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ShellConfig:
    """Shell backend selection."""

    backend: str = "auto"


@dataclass
class AgentBackendConfig:
    """Per-backend agent settings."""

    model: str = "claude-sonnet-4-6"
    web_fetch: bool = False


@dataclass
class AgentConfig:
    """Agent routing and per-backend settings."""

    default: str = "anthropic"
    backends: dict[str, AgentBackendConfig] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Ensure the anthropic backend always has a default entry."""
        if "anthropic" not in self.backends:
            self.backends["anthropic"] = AgentBackendConfig(
                model="claude-sonnet-4-6", web_fetch=True
            )


@dataclass
class ContextConfig:
    """Context provider settings."""

    timeout_ms: int = 200
    providers: list[str] = field(
        default_factory=lambda: ["git", "filesystem", "python", "docker"]
    )


@dataclass
class PermissionRulesConfig:
    """Declarative allow/confirm/deny rules for the permission engine."""

    allow: tuple[str, ...] = ()
    confirm: tuple[str, ...] = ()
    deny: tuple[str, ...] = ()


@dataclass
class Config:
    """Top-level application configuration."""

    shell: ShellConfig = field(default_factory=ShellConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    context: ContextConfig = field(default_factory=ContextConfig)
    permissions: PermissionRulesConfig = field(default_factory=PermissionRulesConfig)


def load_config(path: Path | None = None) -> Config:
    """Load config from path, falling back to defaults for any missing keys."""
    if path is None:
        path = Path.home() / ".config" / "agentsh" / "config.toml"

    if not path.exists():
        return Config()

    with open(path, "rb") as f:
        raw = tomllib.load(f)

    shell = ShellConfig(**raw.get("shell", {}))

    agent_raw = raw.get("agent", {})
    backends: dict[str, AgentBackendConfig] = {}
    for key, val in agent_raw.items():
        if isinstance(val, dict):
            backends[key] = AgentBackendConfig(**val)
    agent = AgentConfig(
        default=agent_raw.get("default", "anthropic"),
        backends=backends,
    )
    agent.__post_init__()

    context_raw = raw.get("context", {})
    context = ContextConfig(
        timeout_ms=context_raw.get("timeout_ms", 200),
        providers=context_raw.get(
            "providers", ["git", "filesystem", "python", "docker"]
        ),
    )

    perm_raw = raw.get("permissions", {}).get("rules", {})
    permissions = PermissionRulesConfig(
        allow=tuple(perm_raw.get("allow", [])),
        confirm=tuple(perm_raw.get("confirm", [])),
        deny=tuple(perm_raw.get("deny", [])),
    )

    return Config(shell=shell, agent=agent, context=context, permissions=permissions)
