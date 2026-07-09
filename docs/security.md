# Security notes

`agentsh` gives an LLM the ability to run shell commands and read/write
files on your behalf. This document describes the mitigations currently
in place, and — just as importantly — where they stop, so you can decide
how to configure `permissions.rules` for your own risk tolerance.

## Permission model

Every call to one of the three built-in tools (`RunCommand`, `ReadFile`,
`WriteFile`) is evaluated by `PermissionEngine.evaluate()`
(`src/agentsh/permissions.py`) against three `fnmatch`-glob rule sets from
config: `deny`, `confirm`, `allow`, checked in that order. A `deny` match
always wins, even against a more specific `allow`/`confirm` rule, and
anything matching no rule at all defaults to `CONFIRM` rather than
`ALLOW` — the system fails closed, not open.

Enforcement happens *inside each tool's own `invoke()`*, not only at the
call sites in the REPL or agent loop. This is deliberate: an earlier
version only checked permissions at the call site
(`agentsh.tools.enforce`), which meant any code path that invoked a tool
directly bypassed the check. Every tool now calls
`self._permissions.enforce(...)` itself, so there is exactly one gate and
no way to route around it by calling a tool a different way.

### fnmatch and shell metacharacters

`allow`/`confirm`/`deny` rules for `RunCommand` are glob patterns matched
against literal command text, e.g. `"RunCommand:git status*"`. A naive
implementation of this is bypassable: an allow-rule like
`"RunCommand:git *"` would also match `git status; rm -rf /`, because
`fnmatch`'s `*` spans every character, including shell chaining and
substitution operators.

`agentsh` mitigates this with `_command_has_shell_metacharacters()`
(`src/agentsh/permissions.py`): any `RunCommand` whose text contains one
of `; & | $` (or a backtick), redirection (`< >`), grouping (`( ) { }`),
an escape character (`\`), `cmd.exe` variable-expansion markers (`% !`),
an embedded newline/carriage return, or a null byte — or that `shlex`
cannot tokenize at all (e.g. unbalanced quotes) — is forced to at least
`CONFIRM`, regardless of whether it matches an `allow` pattern.

**Residual limitation:** this is a fixed blocklist forcing manual
confirmation, not a real shell parser. It's shaped around POSIX/bash
metacharacters; it is defense-in-depth against the common chaining/
substitution bypass, not a guarantee that every way of composing a
dangerous command through `fnmatch` is covered. Write narrow `allow`
rules (exact commands or tight prefixes) rather than broad wildcards if
you want them to mean what they look like they mean.

### Path-based tools and symlinks

`ReadFile`/`WriteFile` permission keys are built from a canonically
resolved, absolute path (`tools/_paths.py:canonical_path`), so a rule like
`"ReadFile:/home/you/project/*"` can't be evaded by a relative path or a
`../` traversal that resolves outside it. An earlier symlink
time-of-check-to-time-of-use (TOCTOU) gap around file tools has also been
closed as part of the same hardening pass that moved enforcement inside
each tool.

## Context and indirect prompt injection

Context providers (`git`, `filesystem`, `docker`, `kubernetes`, `python`,
`history`, `environment`) surface strings that originate outside
`agentsh`'s control and end up in the LLM's system prompt — a git branch
name, a container name, a filesystem entry. Without mitigation, an
attacker who controls one of those strings (e.g. by naming a branch
`"IGNORE ALL PREVIOUS INSTRUCTIONS AND ..."`) could attempt indirect
prompt injection, since the model has no inherent way to distinguish
trusted instructions from untrusted environmental data.

Every context fragment is funneled through `render_context_fragment()`
(`src/agentsh/context/sanitize.py`) before it reaches a prompt, which:

- Escapes every `<` and `>` in the fragment's text (`&lt;`/`&gt;`), so no
  embedded content — including an attempt to spoof the boundary markers
  themselves — can render as real markup.
- Caps each fragment at 4000 characters, truncating with a visible
  marker rather than silently dropping content.
- Wraps the result in `<untrusted-context>` / `</untrusted-context>`
  tags, and the base system prompt (`SYSTEM_PREFIX` in
  `src/agentsh/agent/__init__.py`) instructs the model to treat
  everything inside those tags as inert data, never as instructions.

**Residual limitation:** this stops literal markup/boundary-spoofing
injection, but it is still prompt-level instruction-following, not a hard
technical guarantee. A sufficiently adversarial string can still attempt
semantic injection (content that reads as an instruction without needing
special markup) — whether the model resists that is a property of the
model, not of `agentsh`. Treat any content from an untrusted repo,
container, or filesystem as something the agent will *see*, not
something it's guaranteed to *ignore*.

The `environment` provider additionally applies its own allowlist,
surfacing only a fixed set of generally-safe variable names (`HOME`,
`USER`, `SHELL`, `PATH`, etc.) rather than the full environment, so
secrets stored in other environment variables aren't surfaced into the
prompt in the first place.

## History file hardening

Command history can contain inline secrets — API keys, bearer tokens,
passwords passed as command arguments. Every history file `agentsh`
itself creates (its own REPL history at
`~/.local/share/agentsh/history`, and the bash backend's own history at
`~/.config/agentsh/bash_history`) is opened with an explicit `0o600` mode
(`src/agentsh/history_security.py`), so it's never left world-readable
under a typical `022` umask. A pre-existing file created by an older,
unhardened version of `agentsh` is re-hardened (`fchmod`) the next time
it's written.

This hardening intentionally does **not** extend to a shell's own
native history file (bash's `$HISTFILE`, PowerShell's
`ConsoleHost_history.txt`) — those are only touched at all if you opt in
via `AGENTSH_MIRROR_HISTFILE=1` (see [config.md](config.md)), and mirroring
into them means duplicating potentially-sensitive command text into a
file `agentsh` does not control the permissions of. A one-time warning is
emitted when mirroring is active.

**Windows caveat:** Windows has no `fchmod` and no POSIX permission bits
to harden in the first place, so the re-hardening step is skipped there
rather than raising an error — the `0600` guarantee described above does
not apply on Windows.

## Shell session integrity

The process-backed shell protocol (`src/agentsh/shell/plugin/_base.py`)
detects command completion via a sentinel line printed after each
command. Two properties of that protocol are load-bearing for
correctness, not just style:

- The sentinel includes a per-call nonce, so command output that happens
  to contain a lookalike sentinel string cannot forge command completion
  and desync output parsing from the actual command boundary.
- If a caller times out waiting on an in-flight command (e.g. a context
  provider's collection timeout), the shell is `reset()` — the
  subprocess is killed and restarted — rather than left running, since a
  still-running abandoned command's eventual output would otherwise
  corrupt the parsing of the *next* command issued against that shell.

## Resource limits

Shell and file output is capped (`src/agentsh/limits.py`) both at the
shell-backend level and again inside each tool, so a command or file that
produces unbounded output can't exhaust process memory or blow out the
LLM's context window.

## Known sharp edges (summary)

| Area                                             | Mitigation                                            | What's *not* covered                                                                   |
| ------------------------------------------------ | ----------------------------------------------------- | -------------------------------------------------------------------------------------- |
| Permission rule bypass via chaining/substitution | Metacharacter blocklist forces `CONFIRM`              | Not a full shell grammar parser; write narrow `allow` rules                            |
| Indirect prompt injection via context            | Escaping + boundary tags + model instructions         | Semantic (non-markup) injection is a model-level, not a technical, guarantee           |
| Secrets in history files                         | `0600` on every file `agentsh` creates                | Native shell history files, only touched via opt-in mirroring; no hardening on Windows |
| Symlink/path traversal on file tools             | Canonical path resolution before permission check     | —                                                                                      |
| Shell desync corrupting subsequent commands      | Nonce'd sentinel + reset-on-timeout                   | —                                                                                      |
| Unbounded output exhausting memory/context       | Output capped in the shell backend and again per-tool | —                                                                                      |

If you're deploying `agentsh` somewhere the LLM or its context sources
aren't fully trusted, prefer explicit `deny`/`confirm` rules over broad
`allow` globs, and review `context.providers` for anything that could
surface attacker-influenced strings (e.g. `git`, `docker`) in an
environment where those could be controlled by someone else.
