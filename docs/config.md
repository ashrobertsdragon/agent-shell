# Configuration

`agentsh` reads its config from a single TOML file at:

```text
~/.config/agentsh/config.toml
```

There is currently no environment variable or CLI flag to point at a
different path — `main.py` always calls `load_config()` with no
arguments. If the file doesn't exist, `agentsh` runs entirely on the
defaults below; if it exists, any key you omit falls back to its default
individually (you don't need to repeat the whole schema).

Unknown keys under `[agent]` or `[context]` are a hard error (a
`TypeError` at startup), not a silently-ignored typo — the loader passes
the parsed table straight into the corresponding dataclass constructor.

## Schema

```toml
# ~/.config/agentsh/config.toml

# Which shell backend to drive. "auto" detects from environment
# variables ($SHELL, $PSModulePath, $CMDCMDLINE); set explicitly to skip
# detection or to force a backend detection wouldn't pick.
shell = "auto"  # "auto" | "bash" | "cmd" | "powershell"

[agent]
# Model identifier passed straight through to the provider's SDK.
model = "claude-sonnet-4-6"

# Selects the backend module: agentsh.agent.<provider>.<Provider>Agent
# (e.g. "anthropic" -> agentsh.agent.anthropic.AnthropicAgent). The
# matching optional dependency must be installed
# (uv sync --extra <provider>).
provider = "anthropic"  # "anthropic" | "openai" | "google" | "openrouter"

# Reserved on AgentConfig; present in the schema but not currently
# consumed by any shipped agent backend.
web_fetch = false

# Upper bound passed to the provider's completion call.
max_tokens = 4096

[context]
# Per-provider collection timeout. A provider that doesn't return within
# this window is dropped for that turn (and, for shell-backed providers,
# triggers a shell reset() on the assumption the abandoned command may
# have desynced the session) rather than blocking the REPL.
timeout_ms = 200

# Context providers to run, in order, each turn. Each name resolves to
# agentsh.context.providers.<name>.<Name>Provider.
providers = ["git", "filesystem", "python", "docker", "history", "environment"]

# Declarative permission rules for the three built-in tools. Patterns are
# fnmatch globs matched against a canonical "tool_call_key":
#   RunCommand:<command text, stripped>
#   ReadFile:<absolute POSIX-style path>
#   WriteFile:<absolute POSIX-style path>
#   <ToolName> for anything else
#
# Evaluation order is deny, then confirm, then allow -- a deny match
# always wins regardless of narrower allow/confirm rules. Anything
# matching no rule at all defaults to CONFIRM (never silently allowed).
# A RunCommand whose text contains a shell metacharacter (; & | $ ` < >
# ( ) { } \ % ! embedded newline/null byte) is always forced to at least
# CONFIRM even if it matches an allow pattern -- see docs/security.md.
[permissions.rules]
allow = ["RunCommand:git status*", "RunCommand:ls*", "ReadFile:*"]
confirm = ["RunCommand:git push*"]
deny = ["RunCommand:rm -rf*", "WriteFile:/etc/*"]
```

Note the extra nesting on the permissions table: it's `[permissions.rules]`
in TOML (not a top-level `[permissions]` table with `allow`/`confirm`/
`deny` directly under it) — the loader reads
`raw["permissions"]["rules"]`.

## Field reference

| Key                         | Type                         | Default                                                               | Notes                                          |
| --------------------------- | ---------------------------- | --------------------------------------------------------------------- | ---------------------------------------------- |
| `shell`                     | string                       | `"auto"`                                                              | `"auto"`, `"bash"`, `"cmd"`, or `"powershell"` |
| `agent.model`               | string                       | `"claude-sonnet-4-6"`                                                 | Passed to the provider SDK verbatim            |
| `agent.provider`            | string                       | `"anthropic"`                                                         | Selects `agentsh.agent.<provider>`             |
| `agent.web_fetch`           | bool                         | `false`                                                               | Reserved; not wired to a backend yet           |
| `agent.max_tokens`          | int                          | `4096`                                                                | Passed to the provider SDK                     |
| `context.timeout_ms`        | int                          | `200`                                                                 | Per-provider collection timeout                |
| `context.providers`         | list[string]                 | `["git", "filesystem", "python", "docker", "history", "environment"]` | Order is preserved                             |
| `permissions.rules.allow`   | list[string] (fnmatch globs) | `[]`                                                                  |                                                |
| `permissions.rules.confirm` | list[string] (fnmatch globs) | `[]`                                                                  |                                                |
| `permissions.rules.deny`    | list[string] (fnmatch globs) | `[]`                                                                  | Checked first                                  |

## Environment variables

Configuration is TOML-only, but one behavior is controlled by an
environment variable rather than `config.toml`:

- `AGENTSH_MIRROR_HISTFILE` — when set to a truthy value (`1`, `true`,
  `yes`, or `on`, case-insensitive), the bash backend additionally mirrors
  every command into your shell's native history file (`$HISTFILE`, or
  `~/.bash_history`), for continuity with tools that read your shell's
  own history. This is opt-in because that file is not one `agentsh`
  hardens to mode `0600` — see [security.md](security.md).

## Where else state lives

- `agentsh`'s own command history (used for REPL up-arrow recall) lives
  at `~/.local/share/agentsh/history`, separate from any shell-native
  history file, and is created/maintained with mode `0600`.
- The bash backend additionally reads/writes `~/.config/agentsh/bash_history`
  for its own `history()`/`append_history()` implementation.

Neither of these paths is currently configurable.
