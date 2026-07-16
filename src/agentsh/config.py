"""Config dataclasses and TOML loader for ~/.config/agentsh/config.toml."""

import tomllib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class AgentConfig:
    """Per-agent settings.

    `web_fetch`, when `True`, tells each backend (`agent/{anthropic,
    openai,google,openrouter}.py`) to register that provider's native
    server-side web-fetch/browsing capability on every request. That
    capability runs on the provider's own infrastructure and does not
    go through `agentsh.tools` or `PermissionEngine.evaluate()` --
    enabling it is a documented, intentional bypass of the permission
    engine for outbound web fetches. See docs/security.md for the
    rationale. Defaults to `False`.
    """

    model: str = "claude-sonnet-4-6"
    provider: str = "anthropic"
    web_fetch: bool = False
    max_tokens: int = 4096


@dataclass
class ContextConfig:
    """Context provider settings."""

    timeout_ms: int = 200
    providers: list[str] = field(
        default_factory=lambda: [
            "git",
            "filesystem",
            "python",
            "docker",
            "history",
            "environment",
        ]
    )


@dataclass
class PermissionsConfig:
    """Declarative allow/confirm/deny rules for the permission engine."""

    allow: set[str] = field(default_factory=set)
    confirm: set[str] = field(default_factory=set)
    deny: set[str] = field(default_factory=set)
    write_roots: list[str] = field(default_factory=list)


@dataclass
class Config:
    """Top-level application configuration.

    write_roots is a directory allowlist enforced inside WriteFile itself
    (see tools/write_file.py), independent of the permissions rules
    below: it confines writes even when a rule ALLOWs or an interactive
    CONFIRM approves the call. An empty list (the default) leaves writes
    unconfined, matching prior behavior.
    """

    shell: str = "auto"
    agent: AgentConfig = field(default_factory=AgentConfig)
    context: ContextConfig = field(default_factory=ContextConfig)
    permissions: PermissionsConfig = field(default_factory=PermissionsConfig)


def load_config(path: Path | None = None) -> Config:
    """Load config from path, falling back to defaults for any missing keys."""
    if not path:
        path = Path.home() / ".config" / "agentsh" / "config.toml"

    if not path.exists():
        return Config()

    with open(path, "rb") as f:
        raw = tomllib.load(f)

    shell: str = raw.get("shell", "auto")

    agent_raw: dict = raw.get("agent", {})
    agent = AgentConfig(**agent_raw)

    context_raw: dict = raw.get("context", {})
    context = ContextConfig(**context_raw)

    perm_raw = raw.get("permissions", {})

    write_roots_raw = raw.get("write_roots", [])
    if isinstance(write_roots_raw, str):
        raise ValueError(
            "write_roots must be a list of paths, not a bare string "
            f"(got {write_roots_raw!r}); did you mean [{write_roots_raw!r}]?"
        )
    permissions = PermissionsConfig(
        allow=set(perm_raw.get("allow", [])),
        confirm=set(perm_raw.get("confirm", [])),
        deny=set(perm_raw.get("deny", [])),
        write_roots=write_roots_raw,
    )

    return Config(
        shell=shell,
        agent=agent,
        context=context,
        permissions=permissions,
    )
