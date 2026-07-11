# agentsh

`agentsh` is a shell wrapper that routes each line you type to either a
real shell subprocess (bash, zsh, fish, Nushell, PowerShell, or cmd.exe)
or an LLM agent with tool access to that shell.

Type something that parses as valid shell syntax and it runs exactly as
if you'd typed it into the shell directly — `cd` and working-directory
state behave the same across every backend. Most backends (bash, zsh,
PowerShell, cmd.exe) drive one persistent subprocess per session; fish
and Nushell instead run a fresh process per command (working directory
is still carried over, but environment variable mutations don't
persist between commands — see [Shell backends](#shell-backends)). Type
anything else (or prefix a line with `/agent`) and it's routed to an LLM
agent instead, which sees live environmental context (git branch, cwd,
running containers, ...) and can call tools — `RunCommand`, `ReadFile`,
`WriteFile` — to act on your behalf. Every tool call, from either the
agent or the classifier's own shell routing, passes through a permission
engine that can allow, ask for confirmation, or deny it before it runs.

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) for dependency management
- One of: bash, zsh, fish, Nushell, PowerShell, or cmd.exe on `PATH`
- An API key for at least one supported LLM provider (see below)

## Install

```bash
git clone https://github.com/ashrobertsdragon/agent-shell.git
cd agent-shell
uv sync
```

`agentsh` ships with no LLM provider installed by default — pick the one(s)
you need as an optional extra:

```bash
uv sync --extra anthropic   # Anthropic Claude
uv sync --extra openai      # OpenAI
uv sync --extra google      # Google Gemini
uv sync --extra openrouter  # OpenRouter
```

(Or `uv sync --all-extras` to install all four, e.g. for development.)

## Run

```bash
uv run agentsh
```

This is the same entry point installed as the `agentsh` console script
(`[project.scripts]` in `pyproject.toml`, `agentsh.main:main`). On startup
it:

1. Loads config from `~/.config/agentsh/config.toml` (falling back to
   built-in defaults for anything missing or if the file doesn't exist).
1. Detects your shell (or uses the one configured explicitly) and starts
   it as a persistent backend.
1. Builds the configured context providers and the permission engine.
1. Resolves the configured LLM provider and starts the REPL.

If the configured/detected shell has no backend, or the configured LLM
provider name doesn't resolve to an installed module, `agentsh` exits with
a one-line error rather than a traceback.

## Shell backends

Six shells are currently supported, registered under these names:

| Backend    | Registered name | Detected via                    |
| ---------- | --------------- | ------------------------------- |
| Bash       | `bash`          | `$SHELL` (basename of the path) |
| Zsh        | `zsh`           | `$SHELL` (basename of the path) |
| Fish       | `fish`          | `$SHELL` (basename of the path) |
| Nushell    | `nu`            | `$SHELL` (basename of the path) |
| PowerShell | `powershell`    | `$PSModulePath` present         |
| cmd.exe    | `cmd`           | `$CMDCMDLINE` present           |

With `shell = "auto"` (the default), `agentsh` detects the shell from
environment variables in that order; set `shell` explicitly in
`config.toml` to override detection.

## Permission system

Every tool call (`RunCommand`, `ReadFile`, `WriteFile`) is evaluated
against declarative allow/confirm/deny rules before it runs, in that
order: **deny wins over confirm wins over allow**, so a broad `deny` rule
can never be bypassed by a narrower `allow`/`confirm` rule. Anything that
matches no rule defaults to `CONFIRM` (ask the user).

Rules are `fnmatch` glob patterns matched against a canonical key:

- `RunCommand:<command>` (command text, stripped)
- `ReadFile:<path>` / `WriteFile:<path>` (path resolved to an absolute
  POSIX-style form)
- `<ToolName>` for anything else

Any shell command containing metacharacters — `;`, `&`, `|`, `$`,
backticks, `<`, `>`, `(`, `)`, `{`, `}`, `\`, `%`, `!`, embedded newlines,
or null bytes — is forced to at least `CONFIRM`, even if it matches an
`allow` rule. See [docs/security.md](docs/security.md) for why, and its
residual limits.

See [docs/config.md](docs/config.md) for the full config schema and a
sample `config.toml`.

## Documentation

- [docs/config.md](docs/config.md) — full config schema, defaults, and a
  commented sample `config.toml`
- [docs/security.md](docs/security.md) — permission model, context
  sanitization, history-file hardening, and known sharp edges
- [CONTRIBUTING.md](CONTRIBUTING.md) — how to add a shell backend, a
  context provider, or an agent backend, and the local dev/gate workflow

## Development

```bash
uv sync --all-extras
uv run ruff check .
uv run ruff format --check .
uv run mypy .
uv run pytest -q
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full contributor workflow.
