# Contributing to agentsh

## Dev setup

```bash
git clone https://github.com/ashrobertsdragon/agent-shell.git
cd agent-shell
uv sync --all-extras --group dev
```

`--all-extras` pulls in all four LLM provider SDKs (`anthropic`, `google`,
`openai`, `openrouter`) so the full test suite can run without collection
errors; `--group dev` pulls in `mypy`, `pytest`, `pytest-asyncio`, `ruff`.

## The gate

There is no CI workflow configured in this repository (no
`.github/workflows/`) — the gate below is enforced by convention, not by a
bot, so run it yourself before every commit:

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy .
uv run pytest -q
```

A clean run currently reports `271 passed, 16 skipped` (the skips are
provider-specific tests that no-op without their SDK, plus a handful of
platform-specific cases). If your change adds behavior, add tests for it
— new logic without test coverage will not pass review.

Notes on the toolchain (from `pyproject.toml`):

- **ruff**: `line-length = 80`, target `py312`, docstring convention is
  Google-style (`pydocstyle` via `D` rules). `tests/` and `docs/` get
  relaxed annotation/docstring/line-length rules; `__init__.py` files get
  relaxed `D104`/`E402`.
- **mypy**: strict-ish — `disallow_untyped_defs`, `disallow_untyped_calls`,
  `disallow_any_explicit`, and `warn_return_any` are all on. Every public
  function needs full type annotations; don't reach for `Any`.
- **pytest**: `asyncio_mode = "auto"`, so `async def test_...` works
  without a `@pytest.mark.asyncio` marker. `pythonpath = ["src"]` means
  tests import `agentsh` directly without an editable install.

No code comments — this codebase documents behavior with docstrings
(Google convention), not inline comments. Match that style in anything
you add.

## Architecture in one paragraph

`agentsh` wires everything together in `src/agentsh/main.py:_build_app()`:
a `Shell` backend, a `PermissionEngine`, a `ContextBuilder` (which holds a
list of `ContextProvider`s), an `Agent` backend, and a `ToolRegistry`
holding the three built-in tools (`RunCommand`, `ReadFile`, `WriteFile`).
The REPL (`src/agentsh/repl.py`) classifies each input line as `SHELL` or
`AGENT` (`src/agentsh/classifier.py`) and routes accordingly. Three of the
four pieces above — shells, context providers, and agent backends — are
resolved *dynamically by name* from config, using the same pattern: a
config string picks a module, and a fixed naming convention picks a class
out of that module. That pattern is what makes each of them extensible
without touching a central registry (for context providers and agent
backends) or with a one-line decorator (for shells). The sections below
cover each in turn.

## Adding a shell backend

Shells live in `src/agentsh/shell/plugin/`. Every `.py` file in that
directory (except ones starting with `_`) is imported automatically at
startup by `register_plugins()` (`src/agentsh/shell/plugin/__init__.py`),
so dropping in a new module and decorating its class is enough — there is
no separate registry file to edit.

1. Create `src/agentsh/shell/plugin/<name>.py`.
1. Implement a class satisfying the `Shell` protocol
   (`src/agentsh/shell/protocol.py`):
   `execute`, `cwd` (property), `env`, `history`, `complete`, `can_parse`,
   `render_prompt`, `append_history`, `reset`, `close`. In practice, the
   easiest starting point is subclassing
   `src/agentsh/shell/plugin/_base.py:ProcessBackedShell`, which already
   implements subprocess lifecycle (lazy start, restart-on-desync,
   locked `reset()`/`close()`) — you only need to implement
   `_start_process()` and the shell-specific parts of `execute()`.
1. Decorate the class with `@register("<name>")`
   (`from agentsh.shell._registry import register`). `<name>` is what
   users write as `shell = "<name>"` in `config.toml`, and what
   `detect_shell()` (`src/agentsh/shell/_detect.py`) must return for your
   shell to be auto-detected — extend `_detect.py`'s `_SHELLS` list if you
   want auto-detection support.
1. `can_parse(raw)` and `render_prompt()` have non-functional
   requirements beyond "return the right value":
   - **`can_parse`** is called on every keystroke's worth of REPL input to
     decide shell-vs-agent routing (`classifier.classify`), so it must be
     fast and must never block the event loop — shell out via
     `asyncio.to_thread` (see `bash.py`) or an async subprocess API, with
     a short timeout, and treat a timeout as "not parseable" so a hung
     syntax check can't wedge the REPL.
   - **`render_prompt`** is rendered on every REPL loop iteration and must
     never raise or hang: on any error or timeout, fall back to a plain,
     synchronous string (e.g. `f"{cwd}$ "`) rather than propagating.
   - If your backend uses a sentinel-based protocol to detect command
     completion over a shared stdin/stdout pipe (as `bash.py`/`cmd.py`/
     `powershell.py` do via `_base.py`), the sentinel **must** include a
     per-call nonce (see `new_marker()`) so command output containing a
     lookalike sentinel string can't forge completion, and any timeout
     that abandons an in-flight command must kill/restart the subprocess
     (`reset()`) rather than leave it running — otherwise the next call
     reads stale leftover output from the abandoned command.
1. Write tests under `tests/shell/test_<name>.py` following the existing
   pattern in `tests/shell/test_bash.py` / `test_cmd.py` /
   `test_powershell.py`.

## Adding a context provider

Context providers live in `src/agentsh/context/providers/` and are
resolved dynamically by name — the exact contract, quoted from
`src/agentsh/context/providers/__init__.py`:

> Adding a provider means adding a module here: the module
> `agentsh.context.providers.<name>` must define a class named
> `<Name>Provider`, mirroring how `Agent.from_provider` resolves LLM
> backends.

Concretely:

1. Create `src/agentsh/context/providers/<name>.py` with
   `class <Name>Provider:` (title-cased `<name>` + `Provider`, e.g. a
   `foo` module needs a `FooProvider` class — this is a strict, exact
   naming convention, not a suggestion).
1. Give it a `name: str` class attribute (used as the fragment's
   `provider` field) and implement
   `async def collect(self, shell: Shell) -> ContextFragment | None:`.
   Return `None` when the provider doesn't apply (e.g. `GitProvider`
   returns `None` outside a git repo) — a dropped fragment is not an
   error.
1. Only ever pull information through the `shell` argument (e.g.
   `shell.execute(...)`, `shell.env()`) — providers don't get their own
   ambient access to the system, so tests can fake the shell.
1. If you want the provider active by default, add `"<name>"` to
   `ContextConfig.providers` in `src/agentsh/config.py`; otherwise users
   can opt in via `context.providers` in their own `config.toml`.
1. Every provider's output is treated as **untrusted** and passed through
   `render_context_fragment()` (`src/agentsh/context/sanitize.py`) before
   it reaches an LLM prompt — don't try to build your own prompt
   formatting or escaping in the provider itself; return the raw
   `summary`/`payload` and let the sanitizer handle it. See
   [docs/security.md](docs/security.md) for why this matters.
1. Write tests in `tests/context/test_providers.py` following the
   existing pattern (fake/stub `Shell` instances, assert on the returned
   `ContextFragment` or `None`).

## Adding an agent backend

Agent backends live in `src/agentsh/agent/` and use the same
dynamic-resolution pattern as context providers:
`Agent.from_provider(provider)` (`src/agentsh/agent/base.py`) imports
`agentsh.agent.<provider>` and looks up a class named
`<Provider>Agent` (title-cased), e.g. `agentsh.agent.anthropic` must
define `AnthropicAgent`.

1. Create `src/agentsh/agent/<provider>.py` with `class <Provider>Agent:`.
1. Implement `__init__(self, config: AgentConfig) -> None` (build your
   client from `config.model`, `config.max_tokens`, etc.) and
   `async def respond(self, conversation: list[Message], context: list[ContextFragment], tools: list[SchemaDict]) -> Message`.
1. Message-format contract: `respond()` receives agentsh's canonical
   `Message`/`ToolCall`/`ToolResult` dataclasses
   (`src/agentsh/models.py`) and must return a canonical `Message` back —
   all translation to/from the provider's own wire format happens inside
   your module (see `agent/anthropic.py:_message_to_anthropic` for the
   pattern). Don't leak provider-specific types back to the caller.
1. Build the system prompt from `SYSTEM_PREFIX` (`agentsh.agent`) plus
   each context fragment rendered through
   `render_context_fragment()` (`agentsh.context.sanitize`) — every
   backend must apply the same sanitization/boundary-wrapping, since
   that's what keeps provider-sourced strings (a git branch name, etc.)
   from being interpreted as instructions. See
   [docs/security.md](docs/security.md).
1. If your backend needs a new third-party SDK, add it as a new
   optional-dependency extra in `pyproject.toml`'s
   `[project.optional-dependencies]`, named after your provider string,
   so `uv sync --extra <provider>` installs only what's needed for that
   backend.
1. Write tests in `tests/agent/test_<provider>.py`, following the
   existing provider test modules — mock the SDK client, don't make real
   network calls.

## Adding a tool

Tools (`src/agentsh/tools/`) are the third extension point but are not
name-resolved dynamically — they're registered explicitly in
`main.py:_build_app()` via `app.tools.register(...)`. A tool implements
the `Tool` protocol (`src/agentsh/tools/protocol.py`): `name`,
`description`, `schema` (a `SchemaDict` describing the JSON input schema),
and `async def invoke(self, **kwargs) -> object`. Every tool **must** call
`self._permissions.enforce(...)` at the top of its own `invoke()` — this
is deliberate defense-in-depth so permission enforcement cannot be
bypassed by calling a tool through any path other than the agent loop
(see [docs/security.md](docs/security.md)).

## Pull requests

- One logical change per PR.
- Reference the issue number in the PR description.
- Run the full gate (above) before opening the PR — don't rely on
  reviewers to catch lint/type/test failures.
