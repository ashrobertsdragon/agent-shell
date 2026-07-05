"""Config dataclasses and TOML loader for ~/.config/agentsh/config.toml."""

import tomllib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class AgentConfig:
    """Per-agent settings."""

    model: str = "claude-sonnet-4-6"
    provider: str = "anthropic"
    web_fetch: bool = False
    max_tokens: int = 4096


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

    allow: set[str] = field(default_factory=set)
    confirm: set[str] = field(default_factory=set)
    deny: set[str] = field(default_factory=set)


@dataclass
class Config:
    """Top-level application configuration."""

    shell: str = "auto"
    agent: AgentConfig = field(default_factory=AgentConfig)
    context: ContextConfig = field(default_factory=ContextConfig)
    permissions: PermissionRulesConfig = field(
        default_factory=PermissionRulesConfig
    )


def load_config(path: Path | None = None) -> Config:
    """Load config from path, falling back to defaults for any missing keys."""
    if path is None:
        path = Path.home() / ".config" / "agentsh" / "config.toml"

    if not path.exists():
        return Config()

    with open(path, "rb") as f:
        raw = tomllib.load(f)

    shell: str = raw.get("shell", "auto")

    agent_raw: dict = raw.get("agent", {})
    agent = AgentConfig(**agent_raw)

    context_raw: dict = raw.get("context", {})
    context = ContextConfig(
        timeout_ms=context_raw.get("timeout_ms", 200),
        providers=context_raw.get(
            "providers", ["git", "filesystem", "python", "docker"]
        ),
    )

    perm_raw = raw.get("permissions", {}).get("rules", {})
    permissions = PermissionRulesConfig(
        allow=set(perm_raw.get("allow", [])),
        confirm=set(perm_raw.get("confirm", [])),
        deny=set(perm_raw.get("deny", [])),
    )

    return Config(
        shell=shell, agent=agent, context=context, permissions=permissions
    )
