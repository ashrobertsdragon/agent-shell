# agentsh Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a shell REPL that transparently routes input to either a native Bash subprocess or an Anthropic LLM agent with tool access gated by a declarative permission engine.

**Architecture:** A `prompt_toolkit` REPL renders the shell's own prompt (starship-aware) and appends commands to the user's `$HISTFILE`. Shell input goes directly to a persistent Bash subprocess; natural-language input goes through a classify → context-collect → agent-loop → render pipeline. The agent loop runs until the LLM produces a message with no tool calls, executing each tool call through the permission engine first.

**Tech Stack:** Python 3.12+, `uv` (package manager), `prompt_toolkit` (REPL/UI), `anthropic` SDK, `ruff` + `mypy` (quality), stdlib only for everything else (`asyncio`, `tomllib`, `fnmatch`, `pty`, `tempfile`).

## Global Constraints

- Python 3.12+ — use `match/case`, `str | None` unions, `tomllib` from stdlib.
- Type annotations on every function signature and class attribute.
- Docstrings on all public modules, classes, and functions; no inline comments.
- No `Optional[X]` — use `X | None`.
- `ruff check . && ruff format . && mypy .` must pass before every commit.
- Package layout: `src/agentsh/` with `tests/` at repo root.
- Config file location: `~/.config/agentsh/config.toml`.
- API keys via env vars: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`.

---

## File Map

```text
agentsh/
├── pyproject.toml
├── src/agentsh/
│   ├── __init__.py
│   ├── main.py              # CLI entry point; wires App and calls run_repl
│   ├── app.py               # App dataclass + AppState; pure glue
│   ├── config.py            # Config dataclasses + load_config()
│   ├── models.py            # CommandResult, Message, ToolCall, ToolResult, ContextFragment
│   ├── repl.py              # run_repl(), UI class, prompt/history helpers
│   ├── classifier.py        # InputKind enum + classify()
│   ├── permissions.py       # PermissionLevel, PermissionRules, PermissionEngine
│   ├── events.py            # EventBus + core event dataclasses
│   ├── shell/
│   │   ├── __init__.py
│   │   ├── protocol.py      # Shell Protocol
│   │   └── bash.py          # BashShell
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── protocol.py      # Tool Protocol + ToolRegistry
│   │   ├── run_command.py   # RunCommand tool
│   │   ├── read_file.py     # ReadFile tool
│   │   └── write_file.py    # WriteFile tool (full write + SEARCH/REPLACE patch)
│   ├── context/
│   │   ├── __init__.py
│   │   ├── protocol.py      # ContextProvider Protocol (re-exports ContextFragment)
│   │   ├── builder.py       # ContextBuilder with per-provider timeouts
│   │   └── providers/
│   │       ├── __init__.py
│   │       ├── git.py
│   │       ├── filesystem.py
│   │       ├── python_env.py
│   │       ├── docker.py
│   │       ├── kubernetes.py
│   │       ├── history.py
│   │       └── environment.py
│   └── agent/
│       ├── __init__.py
│       ├── protocol.py      # Agent Protocol + AgentConfig
│       ├── router.py        # AgentRouter (routes to configured default)
│       └── anthropic.py     # AnthropicAgent
└── tests/
    ├── conftest.py
    ├── test_models.py
    ├── test_config.py
    ├── test_classifier.py
    ├── test_permissions.py
    ├── test_events.py
    ├── shell/
    │   └── test_bash.py
    ├── tools/
    │   ├── test_run_command.py
    │   ├── test_read_file.py
    │   └── test_write_file.py
    ├── context/
    │   ├── test_builder.py
    │   └── test_providers.py
    └── agent/
        └── test_anthropic.py
```

---

## Task 1: Project Scaffold

**Files:**

- Create: `pyproject.toml`
- Create: `src/agentsh/__init__.py`
- Create: `tests/conftest.py`

**Interfaces:**

- Produces: `agentsh` package installable via `uv pip install -e .`; `pytest` runs from repo root; `ruff` and `mypy` configured.

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "agentsh"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "prompt-toolkit>=3.0",
    "anthropic>=0.40",
    "openai>=1.50",
]

[project.scripts]
agentsh = "agentsh.main:main"

[tool.hatch.build.targets.wheel]
packages = ["src/agentsh"]

[dependency-groups]
dev = [
    "pytest>=8",
    "pytest-asyncio>=0.24",
    "ruff>=0.9",
    "mypy>=1.13",
    "anthropic>=0.40",
]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]

[tool.mypy]
strict = true
files = ["src", "tests"]

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM"]
```

- [ ] **Step 2: Install dependencies**

```bash
uv sync --all-groups
```

Expected: lockfile created, packages installed.

- [ ] **Step 3: Create package init**

```python
# src/agentsh/__init__.py
"""agentsh — shell wrapper that routes to bash or an LLM agent."""
```

- [ ] **Step 4: Create test conftest**

```python
# tests/conftest.py
"""Shared pytest fixtures."""

import pytest
```

- [ ] **Step 5: Verify tooling runs**

```bash
uv run ruff check . && uv run mypy src/
```

Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml src/ tests/
git commit -m "chore: scaffold agentsh package"
```

---

## Task 2: Core Data Models

**Files:**

- Create: `src/agentsh/models.py`
- Create: `tests/test_models.py`

**Interfaces:**

- Produces: `CommandResult`, `Message`, `ToolCall`, `ToolResult`, `ContextFragment` — imported by every other module.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_models.py
"""Tests for core data models."""

from agentsh.models import CommandResult, ContextFragment, Message, ToolCall, ToolResult


def test_command_result_is_frozen() -> None:
    r = CommandResult(stdout="hi", stderr="", exit_code=0, duration_ms=1.0, cwd="/")
    try:
        r.stdout = "x"  # type: ignore[misc]
        assert False, "should be frozen"
    except Exception:
        pass


def test_message_defaults() -> None:
    m = Message(role="user", content="hello")
    assert m.tool_calls == ()
    assert m.tool_results == ()


def test_tool_result_default_not_error() -> None:
    tr = ToolResult(call_id="abc", content="ok")
    assert not tr.is_error


def test_context_fragment_roundtrip() -> None:
    cf = ContextFragment(
        provider="git", summary="branch main", payload={"branch": "main"}
    )
    assert cf.payload["branch"] == "main"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_models.py -v
```

Expected: `ImportError` or `ModuleNotFoundError`.

- [ ] **Step 3: Implement models**

```python
# src/agentsh/models.py
"""Core data models shared across all agentsh layers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class CommandResult:
    """Result of executing a shell command."""

    stdout: str
    stderr: str
    exit_code: int
    duration_ms: float
    cwd: str


@dataclass(frozen=True, slots=True)
class ToolCall:
    """A tool invocation requested by the LLM."""

    tool_name: str
    arguments: dict[str, Any]
    call_id: str


@dataclass(frozen=True, slots=True)
class ToolResult:
    """The result of executing a ToolCall."""

    call_id: str
    content: str
    is_error: bool = False


@dataclass(frozen=True, slots=True)
class Message:
    """A single turn in the LLM conversation."""

    role: str
    content: str
    tool_calls: tuple[ToolCall, ...] = field(default_factory=tuple)
    tool_results: tuple[ToolResult, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class ContextFragment:
    """A piece of environmental context injected into the agent's system prompt."""

    provider: str
    summary: str
    payload: dict[str, Any]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_models.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Lint and type-check**

```bash
uv run ruff check . && uv run ruff format . && uv run mypy src/
```

- [ ] **Step 6: Commit**

```bash
git add src/agentsh/models.py tests/test_models.py
git commit -m "feat: add core data models"
```

---

## Task 3: Shell Protocol + BashShell

**Files:**

- Create: `src/agentsh/shell/__init__.py`
- Create: `src/agentsh/shell/protocol.py`
- Create: `src/agentsh/shell/bash.py`
- Create: `tests/shell/test_bash.py`
- Create: `tests/shell/__init__.py`

**Interfaces:**

- Consumes: `CommandResult` from `agentsh.models`

- Produces: `Shell` protocol; `BashShell` class; `create_shell()` factory used by `main.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/shell/__init__.py
```

```python
# tests/shell/test_bash.py
"""Integration tests for BashShell against a real bash subprocess."""

import pytest
from agentsh.shell.bash import BashShell


@pytest.fixture
async def shell() -> BashShell:
    s = BashShell()
    yield s
    await s.close()


async def test_execute_echo(shell: BashShell) -> None:
    result = await shell.execute("echo hello")
    assert result.stdout.strip() == "hello"
    assert result.exit_code == 0


async def test_execute_captures_stderr(shell: BashShell) -> None:
    result = await shell.execute("echo err >&2")
    assert "err" in result.stderr
    assert result.exit_code == 0


async def test_execute_tracks_exit_code(shell: BashShell) -> None:
    result = await shell.execute("exit 42 || true; false")
    assert result.exit_code == 1


async def test_execute_tracks_cwd(shell: BashShell) -> None:
    await shell.execute("cd /tmp")
    result = await shell.execute("pwd")
    assert result.stdout.strip() == "/tmp"
    cwd = await shell.cwd()
    assert cwd == "/tmp"


async def test_can_parse_valid(shell: BashShell) -> None:
    assert shell.can_parse("ls -la") is True


async def test_can_parse_invalid(shell: BashShell) -> None:
    assert shell.can_parse(")(invalid((") is False


async def test_can_parse_natural_language(shell: BashShell) -> None:
    assert shell.can_parse("show me all python files") is False


async def test_render_prompt_returns_nonempty(shell: BashShell) -> None:
    prompt = await shell.render_prompt()
    assert isinstance(prompt, str)
    assert len(prompt) > 0


async def test_append_history_writes_to_histfile(
    shell: BashShell, tmp_path: Path
) -> None:
    import os

    histfile = str(tmp_path / ".bash_history")
    with patch.dict(os.environ, {"HISTFILE": histfile}):
        await shell.append_history("ls -la")
    assert "ls -la" in (tmp_path / ".bash_history").read_text()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/shell/test_bash.py -v
```

Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement the Shell protocol**

```python
# src/agentsh/shell/__init__.py
"""Shell abstraction layer."""

from agentsh.shell.bash import BashShell
from agentsh.shell.protocol import Shell

__all__ = ["BashShell", "Shell"]
```

```python
# src/agentsh/shell/protocol.py
"""Shell protocol definition."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from agentsh.models import CommandResult


@runtime_checkable
class Shell(Protocol):
    """Interface for a persistent shell backend."""

    async def execute(self, command: str) -> CommandResult:
        """Execute a command and return its result."""
        ...

    async def cwd(self) -> str:
        """Return the current working directory."""
        ...

    async def env(self) -> dict[str, str]:
        """Return the current environment variables."""
        ...

    async def history(self, limit: int = 100) -> list[str]:
        """Return recent command history entries."""
        ...

    async def complete(self, partial: str) -> list[str]:
        """Return completions for a partial command string."""
        ...

    def can_parse(self, raw: str) -> bool:
        """Return True if raw is valid shell syntax."""
        ...

    async def render_prompt(self) -> str:
        """Return the rendered shell prompt string as the user would see it."""
        ...

    async def append_history(self, command: str) -> None:
        """Append a command to the shell's persistent history store."""
        ...

    async def close(self) -> None:
        """Terminate the underlying subprocess."""
        ...
```

**History file locations by shell (for implementers of Phase 5 backends):**

- Bash/Zsh: `$HISTFILE` env var, default `~/.bash_history` / `~/.zsh_history`

- PowerShell (Linux/Mac): `~/.local/share/powershell/PSReadLine/ConsoleHost_history.txt`

- PowerShell (Windows): `%APPDATA%\Microsoft\Windows\PowerShell\PSReadLine\ConsoleHost_history.txt`

- CMD: `~/.clink_history` if clink is installed; otherwise CMD has no persistent history.

- [ ] **Step 4: Implement BashShell**

```python
# src/agentsh/shell/bash.py
"""Persistent Bash shell backend."""

from __future__ import annotations

import asyncio
import os
import subprocess
import tempfile
import time
from pathlib import Path

from agentsh.models import CommandResult

_SENTINEL = "__AGENTSH_DONE_8675309__"


class BashShell:
    """Wraps a persistent bash subprocess; tracks cwd after every command."""

    _process: asyncio.subprocess.Process | None
    _cwd: str
    _lock: asyncio.Lock

    def __init__(self) -> None:
        self._process = None
        self._cwd = os.getcwd()
        self._lock = asyncio.Lock()

    async def _ensure_started(self) -> asyncio.subprocess.Process:
        if self._process is None or self._process.returncode is not None:
            self._process = await asyncio.create_subprocess_exec(
                "bash",
                "--noprofile",
                "--norc",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
        return self._process

    async def execute(self, command: str) -> CommandResult:
        """Execute a shell command and return stdout, stderr, exit code, and updated cwd."""
        async with self._lock:
            proc = await self._ensure_started()
            assert proc.stdin and proc.stdout

            fd, stderr_path = tempfile.mkstemp(prefix="agentsh_stderr_")
            os.close(fd)

            start = time.monotonic()
            wrapped = (
                f"({command}) 2>{stderr_path}\n"
                f"__ec__=$?\n"
                f'printf "%s:%d:%s\\n" "{_SENTINEL}" "$__ec__" "$(pwd)"\n'
            )
            proc.stdin.write(wrapped.encode())
            await proc.stdin.drain()

            chunks: list[str] = []
            exit_code = 1
            async for line in proc.stdout:
                decoded = line.decode(errors="replace")
                if decoded.startswith(f"{_SENTINEL}:"):
                    _, code_str, cwd = decoded.strip().split(":", 2)
                    exit_code = int(code_str)
                    self._cwd = cwd
                    break
                chunks.append(decoded)

            stderr_content = Path(stderr_path).read_text(errors="replace")
            Path(stderr_path).unlink(missing_ok=True)
            duration_ms = (time.monotonic() - start) * 1000

            return CommandResult(
                stdout="".join(chunks),
                stderr=stderr_content,
                exit_code=exit_code,
                duration_ms=duration_ms,
                cwd=self._cwd,
            )

    async def cwd(self) -> str:
        """Return the last tracked working directory."""
        return self._cwd

    async def env(self) -> dict[str, str]:
        """Return the subprocess environment by running `env`."""
        result = await self.execute("env")
        env: dict[str, str] = {}
        for line in result.stdout.splitlines():
            if "=" in line:
                k, _, v = line.partition("=")
                env[k] = v
        return env

    async def history(self, limit: int = 100) -> list[str]:
        """Return lines from $HISTFILE (default ~/.bash_history)."""
        histfile = os.environ.get("HISTFILE", str(Path.home() / ".bash_history"))
        try:
            lines = Path(histfile).read_text().splitlines()
            return lines[-limit:]
        except FileNotFoundError:
            return []

    async def complete(self, partial: str) -> list[str]:
        """Return up to 20 command completions via compgen."""
        result = await self.execute(f"compgen -c {partial!r} 2>/dev/null | head -20")
        return result.stdout.splitlines()

    def can_parse(self, raw: str) -> bool:
        """Return True if bash -n accepts the input as valid syntax."""
        try:
            result = subprocess.run(
                ["bash", "-n", "-c", raw],
                capture_output=True,
                timeout=1.0,
            )
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            return False

    async def render_prompt(self) -> str:
        """Evaluate PS1 via bash -i so the user's .bashrc, starship, etc. are active.

        bash -i sources .bashrc (interactive mode). ${PS1@P} expands all
        bash prompt sequences including command substitutions. \\001/\\002
        are readline non-printing markers that must be stripped before display.
        """
        import shlex

        try:
            result = subprocess.run(
                [
                    "bash",
                    "-i",
                    "-c",
                    f"cd {shlex.quote(self._cwd)} && printf '%s' \"${{PS1@P}}\"",
                ],
                capture_output=True,
                text=True,
                timeout=2.0,
                env=os.environ,
            )
            prompt = result.stdout.replace("\001", "").replace("\002", "")
            if prompt:
                return prompt
        except subprocess.TimeoutExpired:
            pass
        return f"{self._cwd}$ "

    async def append_history(self, command: str) -> None:
        """Append command to $HISTFILE (bash history file)."""
        histfile = os.environ.get("HISTFILE", str(Path.home() / ".bash_history"))
        try:
            with open(histfile, "a") as f:
                f.write(command + "\n")
        except OSError:
            pass

    async def close(self) -> None:
        """Terminate the underlying bash subprocess."""
        if self._process and self._process.returncode is None:
            self._process.terminate()
            await self._process.wait()
```

- [ ] **Step 5: Run tests**

```bash
uv run pytest tests/shell/test_bash.py -v
```

Expected: all pass. The cwd test relies on tracking across two `execute` calls.

- [ ] **Step 6: Lint and type-check**

```bash
uv run ruff check . && uv run ruff format . && uv run mypy src/
```

- [ ] **Step 7: Commit**

```bash
git add src/agentsh/shell/ tests/shell/
git commit -m "feat: add Shell protocol and BashShell"
```

---

## Task 4: Tool Protocol + ToolRegistry + RunCommand

**Files:**

- Create: `src/agentsh/tools/__init__.py`
- Create: `src/agentsh/tools/protocol.py`
- Create: `src/agentsh/tools/run_command.py`
- Create: `tests/tools/__init__.py`
- Create: `tests/tools/test_run_command.py`

**Interfaces:**

- Consumes: `Shell` from `agentsh.shell`, `CommandResult` from `agentsh.models`
- Produces: `Tool` protocol; `ToolRegistry`; `RunCommand` tool — used by the REPL and agentic loop.

**Note:** `RunCommand` receives a `PermissionEngine` dependency but at this task that engine is not yet built. Pass a `None` sentinel and skip permission checking until Task 8.

- [ ] **Step 1: Write failing tests**

```python
# tests/tools/__init__.py
```

```python
# tests/tools/test_run_command.py
"""Tests for the RunCommand tool."""

import pytest
from unittest.mock import AsyncMock
from agentsh.models import CommandResult
from agentsh.tools.run_command import RunCommand
from agentsh.tools.protocol import ToolRegistry


@pytest.fixture
def mock_shell() -> AsyncMock:
    shell = AsyncMock()
    shell.execute = AsyncMock(
        return_value=CommandResult(
            stdout="hello\n", stderr="", exit_code=0, duration_ms=5.0, cwd="/tmp"
        )
    )
    return shell


async def test_run_command_invokes_shell(mock_shell: AsyncMock) -> None:
    tool = RunCommand(shell=mock_shell, permissions=None)
    result = await tool.invoke(command="echo hello")
    mock_shell.execute.assert_called_once_with("echo hello")
    assert result.stdout == "hello\n"


def test_tool_registry_get() -> None:
    registry = ToolRegistry()
    tool = AsyncMock()
    tool.name = "RunCommand"
    registry.register(tool)
    assert registry.get("RunCommand") is tool


def test_tool_registry_schemas() -> None:
    registry = ToolRegistry()
    tool = AsyncMock()
    tool.name = "RunCommand"
    tool.schema = {"name": "RunCommand"}
    registry.register(tool)
    assert registry.schemas() == [{"name": "RunCommand"}]
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/tools/test_run_command.py -v
```

Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement Tool protocol and ToolRegistry**

```python
# src/agentsh/tools/__init__.py
"""Tool layer — runnable actions available to the agent and REPL."""

from agentsh.tools.protocol import Tool, ToolRegistry
from agentsh.tools.run_command import RunCommand

__all__ = ["RunCommand", "Tool", "ToolRegistry"]
```

```python
# src/agentsh/tools/protocol.py
"""Tool protocol and ToolRegistry."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Tool(Protocol):
    """Interface for an agent-callable tool."""

    name: str
    description: str
    schema: dict[str, Any]

    async def invoke(self, **kwargs: Any) -> Any:
        """Execute the tool with the given arguments."""
        ...


class ToolRegistry:
    """Registry mapping tool names to Tool instances."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool, overwriting any existing tool with the same name."""
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        """Return a tool by name; raises KeyError if not found."""
        return self._tools[name]

    def schemas(self) -> list[dict[str, Any]]:
        """Return the JSON schema for every registered tool."""
        return [t.schema for t in self._tools.values()]
```

- [ ] **Step 4: Implement RunCommand**

```python
# src/agentsh/tools/run_command.py
"""RunCommand tool — executes arbitrary shell commands."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agentsh.models import CommandResult
from agentsh.shell.protocol import Shell

if TYPE_CHECKING:
    from agentsh.permissions import PermissionEngine


class RunCommand:
    """Executes a shell command through the Shell backend.

    Permission checking is delegated to the injected PermissionEngine.
    When permissions is None, all commands are allowed (used before
    the permission engine is wired in).
    """

    name = "RunCommand"
    description = (
        "Execute a shell command and return its stdout, stderr, and exit code."
    )
    schema: dict[str, Any] = {
        "name": "RunCommand",
        "description": "Execute a shell command and return its stdout, stderr, and exit code.",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute verbatim.",
                }
            },
            "required": ["command"],
        },
    }

    def __init__(self, shell: Shell, permissions: "PermissionEngine | None") -> None:
        self._shell = shell
        self._permissions = permissions

    async def invoke(self, **kwargs: Any) -> CommandResult:
        """Execute the given command string through the shell."""
        command: str = kwargs["command"]
        return await self._shell.execute(command)
```

- [ ] **Step 5: Run tests**

```bash
uv run pytest tests/tools/test_run_command.py -v
```

Expected: all pass.

- [ ] **Step 6: Lint and type-check**

```bash
uv run ruff check . && uv run ruff format . && uv run mypy src/
```

- [ ] **Step 7: Commit**

```bash
git add src/agentsh/tools/ tests/tools/
git commit -m "feat: add Tool protocol, ToolRegistry, and RunCommand"
```

---

## Task 5: Minimal REPL (Shell Path Only)

**Files:**

- Create: `src/agentsh/repl.py`
- Create: `src/agentsh/app.py`
- Create: `src/agentsh/main.py`

**Interfaces:**

- Consumes: `BashShell`, `ToolRegistry`, `RunCommand` from earlier tasks
- Produces: `run_repl(app)` coroutine; `App` dataclass; `main()` entry point. At this stage the REPL only handles the shell path.

**Prompt strategy:** Delegate entirely to `shell.render_prompt()`. The REPL makes no decisions about prompt format — that is the shell's concern.

**History strategy:** Delegate entirely to `shell.append_history()`. Each shell backend knows where its history lives.

- [ ] **Step 1: Create AppState and App**

```python
# src/agentsh/app.py
"""App — the top-level wiring object; holds all runtime dependencies."""

from __future__ import annotations

from dataclasses import dataclass, field

from agentsh.models import Message
from agentsh.shell.protocol import Shell
from agentsh.tools.protocol import ToolRegistry


@dataclass
class AppState:
    """Mutable runtime state shared across REPL turns."""

    conversation: list[Message] = field(default_factory=list)


@dataclass
class App:
    """Dependency container; constructed in main.py and passed to run_repl."""

    shell: Shell
    tools: ToolRegistry
    state: AppState
```

- [ ] **Step 2: Implement the REPL**

```python
# src/agentsh/repl.py
"""REPL loop and UI helpers."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import ANSI
from prompt_toolkit.history import FileHistory

from agentsh.app import App
from agentsh.models import CommandResult, Message


def _render(result: CommandResult | Message) -> None:
    """Print a CommandResult or Message to stdout/stderr."""
    match result:
        case CommandResult():
            if result.stdout:
                print(result.stdout, end="")
            if result.stderr:
                print(result.stderr, end="", file=sys.stderr)
        case Message():
            if result.content:
                print(result.content)


async def run_repl(app: App) -> None:
    """Run the main REPL loop until EOF or KeyboardInterrupt."""
    history_dir = Path.home() / ".local" / "share" / "agentsh"
    history_dir.mkdir(parents=True, exist_ok=True)
    session: PromptSession[str] = PromptSession(
        history=FileHistory(str(history_dir / "history"))
    )

    while True:
        try:
            prompt = await app.shell.render_prompt()
            raw: str = await session.prompt_async(ANSI(prompt))
        except EOFError:
            break
        except KeyboardInterrupt:
            continue

        raw = raw.strip()
        if not raw:
            continue

        await app.shell.append_history(raw)
        result = await app.tools.get("RunCommand").invoke(command=raw)
        _render(result)
```

- [ ] **Step 3: Create the entry point**

```python
# src/agentsh/main.py
"""CLI entry point."""

from __future__ import annotations

import asyncio

from agentsh.app import App, AppState
from agentsh.repl import run_repl
from agentsh.shell.bash import BashShell
from agentsh.tools.protocol import ToolRegistry
from agentsh.tools.run_command import RunCommand


def _build_app() -> App:
    shell = BashShell()
    tools = ToolRegistry()
    tools.register(RunCommand(shell=shell, permissions=None))
    return App(shell=shell, tools=tools, state=AppState())


def main() -> None:
    """Entry point for the agentsh CLI."""
    app = _build_app()
    asyncio.run(run_repl(app))
```

- [ ] **Step 4: Run it manually**

```bash
uv run agentsh
```

Expected: A prompt appears (starship or PS1 fallback). Type `echo hello` — it executes. Type `cd /tmp` and verify the prompt updates. `Ctrl+D` exits.

- [ ] **Step 5: Lint and type-check**

```bash
uv run ruff check . && uv run ruff format . && uv run mypy src/
```

- [ ] **Step 6: Commit**

```bash
git add src/agentsh/repl.py src/agentsh/app.py src/agentsh/main.py
git commit -m "feat: add minimal shell REPL with prompt_toolkit and history"
```

---

## Task 6: Config Loading

**Files:**

- Create: `src/agentsh/config.py`
- Create: `tests/test_config.py`

**Interfaces:**

- Produces: `Config`, `ShellConfig`, `AgentBackendConfig`, `ContextConfig`, `PermissionRulesConfig` dataclasses; `load_config() -> Config` function used by `main.py`.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_config.py
"""Tests for config loading."""

import textwrap
from pathlib import Path
import pytest
from agentsh.config import Config, load_config


def test_load_defaults_when_no_file(tmp_path: Path) -> None:
    cfg = load_config(tmp_path / "nonexistent.toml")
    assert cfg.agent.default == "anthropic"
    assert cfg.shell.backend == "auto"


def test_load_overrides_from_file(tmp_path: Path) -> None:
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
    cfg = load_config(tmp_path / "no.toml")
    assert cfg.permissions.allow == ()
    assert cfg.permissions.confirm == ()
    assert cfg.permissions.deny == ()
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/test_config.py -v
```

- [ ] **Step 3: Implement config**

```python
# src/agentsh/config.py
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
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_config.py -v
```

Expected: all pass.

- [ ] **Step 5: Lint and type-check**

```bash
uv run ruff check . && uv run ruff format . && uv run mypy src/
```

- [ ] **Step 6: Commit**

```bash
git add src/agentsh/config.py tests/test_config.py
git commit -m "feat: add config loading from TOML"
```

---

## Task 7: Permission Engine

**Files:**

- Create: `src/agentsh/permissions.py`
- Create: `tests/test_permissions.py`

**Interfaces:**

- Consumes: `PermissionRulesConfig` from `agentsh.config`

- Produces: `PermissionLevel`, `PermissionEngine` — used by `RunCommand` and the agentic loop's confirm step.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_permissions.py
"""Tests for the permission engine, including deny-precedence."""

import pytest
from agentsh.config import PermissionRulesConfig
from agentsh.permissions import PermissionEngine, PermissionLevel


@pytest.fixture
def engine() -> PermissionEngine:
    rules = PermissionRulesConfig(
        allow=("RunCommand:ls*", "RunCommand:pwd", "ReadFile"),
        confirm=("RunCommand:git commit*", "WriteFile"),
        deny=("RunCommand:rm -rf*",),
    )
    return PermissionEngine(rules)


def test_allow(engine: PermissionEngine) -> None:
    assert engine.evaluate("RunCommand:ls -la") == PermissionLevel.ALLOW


def test_confirm(engine: PermissionEngine) -> None:
    assert engine.evaluate("RunCommand:git commit -m 'x'") == PermissionLevel.CONFIRM


def test_deny(engine: PermissionEngine) -> None:
    assert engine.evaluate("RunCommand:rm -rf /") == PermissionLevel.DENY


def test_deny_beats_allow() -> None:
    rules = PermissionRulesConfig(
        allow=("RunCommand:rm*",),
        deny=("RunCommand:rm -rf*",),
    )
    engine = PermissionEngine(rules)
    assert engine.evaluate("RunCommand:rm -rf /tmp/x") == PermissionLevel.DENY


def test_default_is_confirm(engine: PermissionEngine) -> None:
    assert engine.evaluate("RunCommand:unknown-tool") == PermissionLevel.CONFIRM


def test_read_file_allow(engine: PermissionEngine) -> None:
    assert engine.evaluate("ReadFile") == PermissionLevel.ALLOW


def test_write_file_confirm(engine: PermissionEngine) -> None:
    assert engine.evaluate("WriteFile") == PermissionLevel.CONFIRM
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/test_permissions.py -v
```

- [ ] **Step 3: Implement PermissionEngine**

```python
# src/agentsh/permissions.py
"""Permission engine — evaluates tool call keys against allow/confirm/deny rules."""

from __future__ import annotations

from enum import auto, Enum
from fnmatch import fnmatch

from agentsh.config import PermissionRulesConfig


class PermissionLevel(Enum):
    """Outcome of a permission evaluation."""

    ALLOW = auto()
    CONFIRM = auto()
    DENY = auto()


class PermissionEngine:
    """Evaluates a tool_call_key against declarative fnmatch rules.

    Deny is checked first so a broad deny can never be overridden by a
    narrower allow or confirm rule.

    The tool_call_key format is:
      - ``"{tool_name}:{command}"`` for RunCommand
      - ``"{tool_name}"`` for ReadFile / WriteFile
    """

    def __init__(self, rules: PermissionRulesConfig) -> None:
        self._rules = rules

    def evaluate(self, tool_call_key: str) -> PermissionLevel:
        """Return the permission level for the given tool call key."""
        if any(fnmatch(tool_call_key, p) for p in self._rules.deny):
            return PermissionLevel.DENY
        if any(fnmatch(tool_call_key, p) for p in self._rules.confirm):
            return PermissionLevel.CONFIRM
        if any(fnmatch(tool_call_key, p) for p in self._rules.allow):
            return PermissionLevel.ALLOW
        return PermissionLevel.CONFIRM
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_permissions.py -v
```

Expected: all pass.

- [ ] **Step 5: Lint and type-check**

```bash
uv run ruff check . && uv run ruff format . && uv run mypy src/
```

- [ ] **Step 6: Commit**

```bash
git add src/agentsh/permissions.py tests/test_permissions.py
git commit -m "feat: add permission engine with deny-first rule evaluation"
```

---

## Task 8: Wire Permissions into RunCommand + REPL

**Files:**

- Modify: `src/agentsh/tools/run_command.py`
- Modify: `src/agentsh/repl.py`
- Modify: `src/agentsh/app.py`
- Modify: `src/agentsh/main.py`
- Modify: `tests/tools/test_run_command.py`

**Interfaces:**

- Consumes: `PermissionEngine`, `PermissionLevel` from `agentsh.permissions`

- Produces: RunCommand enforces DENY by raising; CONFIRM prompts via `UI.confirm()`; ALLOW passes through.

- [ ] **Step 1: Write failing test for deny**

Add to `tests/tools/test_run_command.py`:

```python
from agentsh.config import PermissionRulesConfig
from agentsh.permissions import PermissionEngine
from agentsh.tools.run_command import PermissionDeniedError


async def test_run_command_deny_raises(mock_shell: AsyncMock) -> None:
    rules = PermissionRulesConfig(deny=("RunCommand:rm*",))
    permissions = PermissionEngine(rules)
    tool = RunCommand(shell=mock_shell, permissions=permissions)
    with pytest.raises(PermissionDeniedError):
        await tool.invoke(command="rm -rf /")
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/tools/test_run_command.py::test_run_command_deny_raises -v
```

- [ ] **Step 3: Update RunCommand to enforce DENY**

Replace `src/agentsh/tools/run_command.py` with:

```python
# src/agentsh/tools/run_command.py
"""RunCommand tool — executes arbitrary shell commands with permission gating."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agentsh.models import CommandResult
from agentsh.shell.protocol import Shell

if TYPE_CHECKING:
    from agentsh.permissions import PermissionEngine, PermissionLevel


class PermissionDeniedError(Exception):
    """Raised when a command is blocked by a DENY permission rule."""


class RunCommand:
    """Executes a shell command through the Shell backend.

    When a PermissionEngine is provided:
    - DENY: raises PermissionDeniedError immediately.
    - CONFIRM: the caller (REPL / agentic loop) must prompt before calling invoke().
    - ALLOW: passes through without prompting.
    When permissions is None, all commands are allowed.
    """

    name = "RunCommand"
    description = (
        "Execute a shell command and return its stdout, stderr, and exit code."
    )
    schema: dict[str, Any] = {
        "name": "RunCommand",
        "description": "Execute a shell command and return its stdout, stderr, and exit code.",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute verbatim.",
                }
            },
            "required": ["command"],
        },
    }

    def __init__(self, shell: Shell, permissions: "PermissionEngine | None") -> None:
        self._shell = shell
        self._permissions = permissions

    def _check_key(self, command: str) -> "PermissionLevel | None":
        """Return the permission level for the command, or None if no engine."""
        if self._permissions is None:
            return None
        from agentsh.permissions import PermissionLevel

        key = f"RunCommand:{command}"
        return self._permissions.evaluate(key)

    async def invoke(self, **kwargs: Any) -> CommandResult:
        """Execute the given command, raising PermissionDeniedError if denied."""
        from agentsh.permissions import PermissionLevel

        command: str = kwargs["command"]
        level = self._check_key(command)
        if level == PermissionLevel.DENY:
            raise PermissionDeniedError(f"Command denied by policy: {command}")
        return await self._shell.execute(command)
```

- [ ] **Step 4: Add `UI` class and `confirm()` to repl.py**

Add a `UI` class to `src/agentsh/repl.py` and thread a session into it. Replace the existing `_render` function with a method and add `confirm`:

In `src/agentsh/repl.py`, add after imports:

```python
from agentsh.models import CommandResult, Message, ToolCall
from agentsh.permissions import PermissionLevel


class UI:
    """Handles user-facing I/O: prompting, rendering results, and confirmations."""

    def __init__(self, session: "PromptSession[str]") -> None:
        self._session = session

    def render(self, result: CommandResult | Message) -> None:
        """Print a result to stdout (or stderr for command stderr)."""
        match result:
            case CommandResult():
                if result.stdout:
                    print(result.stdout, end="")
                if result.stderr:
                    print(result.stderr, end="", file=sys.stderr)
            case Message():
                if result.content:
                    print(result.content)

    async def confirm(self, tool_name: str, arguments: dict[str, Any]) -> bool:
        """Prompt the user to allow or deny a CONFIRM-level tool call."""
        label = arguments.get("command") or arguments.get("path") or tool_name
        print(f"\n[agentsh] permission required — {tool_name}: {label}")
        try:
            answer = await self._session.prompt_async("Allow? [y/N] ")
            return answer.strip().lower() == "y"
        except (EOFError, KeyboardInterrupt):
            return False
```

Update `App` to hold a `UI` and `PermissionEngine`:

```python
# src/agentsh/app.py
"""App — the top-level wiring object; holds all runtime dependencies."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from agentsh.models import Message
from agentsh.shell.protocol import Shell
from agentsh.tools.protocol import ToolRegistry

if TYPE_CHECKING:
    from agentsh.permissions import PermissionEngine
    from agentsh.repl import UI


@dataclass
class AppState:
    """Mutable runtime state shared across REPL turns."""

    conversation: list[Message] = field(default_factory=list)


@dataclass
class App:
    """Dependency container; constructed in main.py and passed to run_repl."""

    shell: Shell
    tools: ToolRegistry
    permissions: "PermissionEngine"
    state: AppState
    ui: "UI | None" = None
```

Update `run_repl` in `repl.py` to create `UI`, attach it to `app`, and handle `PermissionDeniedError` in the shell path:

```python
async def run_repl(app: App) -> None:
    """Run the main REPL loop until EOF or KeyboardInterrupt."""
    from agentsh.permissions import PermissionLevel
    from agentsh.tools.run_command import PermissionDeniedError

    history_dir = Path.home() / ".local" / "share" / "agentsh"
    history_dir.mkdir(parents=True, exist_ok=True)
    session: PromptSession[str] = PromptSession(
        history=FileHistory(str(history_dir / "history"))
    )
    ui = UI(session)
    app.ui = ui

    while True:
        try:
            prompt = await app.shell.render_prompt()
            raw: str = await session.prompt_async(ANSI(prompt))
        except EOFError:
            break
        except KeyboardInterrupt:
            continue

        raw = raw.strip()
        if not raw:
            continue

        await app.shell.append_history(raw)

        try:
            run_cmd = app.tools.get("RunCommand")
            from agentsh.permissions import PermissionEngine

            assert isinstance(app.permissions, PermissionEngine)
            key = f"RunCommand:{raw}"
            level = app.permissions.evaluate(key)
            if level == PermissionLevel.DENY:
                print(f"[agentsh] denied: {raw}", file=sys.stderr)
                continue
            if level == PermissionLevel.CONFIRM:
                if not await ui.confirm("RunCommand", {"command": raw}):
                    print("[agentsh] cancelled.", file=sys.stderr)
                    continue
            result = await run_cmd.invoke(command=raw)
            ui.render(result)
        except PermissionDeniedError as e:
            print(f"[agentsh] {e}", file=sys.stderr)
```

Update `main.py` to wire `PermissionEngine`:

```python
# src/agentsh/main.py
"""CLI entry point."""

from __future__ import annotations

import asyncio

from agentsh.app import App, AppState
from agentsh.config import load_config
from agentsh.permissions import PermissionEngine
from agentsh.repl import run_repl
from agentsh.shell.bash import BashShell
from agentsh.tools.protocol import ToolRegistry
from agentsh.tools.run_command import RunCommand


def _build_app() -> App:
    config = load_config()
    shell = BashShell()
    permissions = PermissionEngine(config.permissions)
    tools = ToolRegistry()
    tools.register(RunCommand(shell=shell, permissions=permissions))
    return App(shell=shell, tools=tools, permissions=permissions, state=AppState())


def main() -> None:
    """Entry point for the agentsh CLI."""
    app = _build_app()
    asyncio.run(run_repl(app))
```

- [ ] **Step 5: Run all tests**

```bash
uv run pytest -v
```

Expected: all pass.

- [ ] **Step 6: Lint and type-check**

```bash
uv run ruff check . && uv run ruff format . && uv run mypy src/
```

- [ ] **Step 7: Commit**

```bash
git add src/agentsh/ tests/
git commit -m "feat: wire permission engine into RunCommand and REPL"
```

---

## Task 9: Classifier

**Files:**

- Create: `src/agentsh/classifier.py`
- Create: `tests/test_classifier.py`

**Interfaces:**

- Consumes: `Shell` protocol from `agentsh.shell`

- Produces: `InputKind` enum; `classify(raw, shell) -> InputKind` function — used by `run_repl`.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_classifier.py
"""Table-driven tests for the input classifier."""

import pytest
from unittest.mock import MagicMock
from agentsh.classifier import InputKind, classify


@pytest.fixture
def shell() -> MagicMock:
    s = MagicMock()
    s.can_parse = MagicMock(
        side_effect=lambda raw: raw.startswith(("ls", "cd", "echo", "git", "pwd"))
    )
    return s


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("/agent tell me a joke", InputKind.AGENT),
        ("ls -la", InputKind.SHELL_PARSEABLE),
        ("echo hello world", InputKind.SHELL_PARSEABLE),
        ("show me all python files", InputKind.AGENT),
        ("what does this repo do", InputKind.AGENT),
        ("/agent ", InputKind.AGENT),
    ],
)
def test_classify(raw: str, expected: InputKind, shell: MagicMock) -> None:
    assert classify(raw, shell) == expected
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/test_classifier.py -v
```

- [ ] **Step 3: Implement classifier**

```python
# src/agentsh/classifier.py
"""Input classifier — decides whether to route input to shell or agent."""

from __future__ import annotations

from enum import auto, Enum

from agentsh.shell.protocol import Shell


class InputKind(Enum):
    """The result of classifying a raw user input string."""

    AGENT = auto()
    SHELL_PARSEABLE = auto()


def classify(raw: str, shell: Shell) -> InputKind:
    """Return AGENT or SHELL_PARSEABLE based on input content."""
    match raw:
        case s if s.startswith("/agent "):
            return InputKind.AGENT
        case s if shell.can_parse(s):
            return InputKind.SHELL_PARSEABLE
        case _:
            return InputKind.AGENT
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_classifier.py -v
```

- [ ] **Step 5: Lint and type-check**

```bash
uv run ruff check . && uv run ruff format . && uv run mypy src/
```

- [ ] **Step 6: Commit**

```bash
git add src/agentsh/classifier.py tests/test_classifier.py
git commit -m "feat: add input classifier"
```

---

## Task 10: Context Providers + ContextBuilder

**Files:**

- Create: `src/agentsh/context/__init__.py`
- Create: `src/agentsh/context/protocol.py`
- Create: `src/agentsh/context/builder.py`
- Create: `src/agentsh/context/providers/__init__.py`
- Create: `src/agentsh/context/providers/git.py`
- Create: `src/agentsh/context/providers/filesystem.py`
- Create: `tests/context/__init__.py`
- Create: `tests/context/test_builder.py`
- Create: `tests/context/test_providers.py`

**Interfaces:**

- Consumes: `Shell`, `ContextFragment` from `agentsh.models`

- Produces: `ContextProvider` protocol; `ContextBuilder` with per-provider timeouts; `GitProvider`, `FilesystemProvider` — consumed by `AnthropicAgent`.

- [ ] **Step 1: Write failing tests**

```python
# tests/context/__init__.py
```

```python
# tests/context/test_builder.py
"""Tests for ContextBuilder timeout and failure isolation."""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock
from agentsh.context.builder import ContextBuilder
from agentsh.models import ContextFragment


@pytest.fixture
def shell() -> MagicMock:
    return MagicMock()


async def test_builder_collects_fragments(shell: MagicMock) -> None:
    frag = ContextFragment(provider="test", summary="test", payload={})
    provider = MagicMock()
    provider.collect = AsyncMock(return_value=frag)
    builder = ContextBuilder(providers=[provider], timeout_ms=200)
    result = await builder.build(shell)
    assert result == [frag]


async def test_builder_swallows_failures(shell: MagicMock) -> None:
    provider = MagicMock()
    provider.collect = AsyncMock(side_effect=RuntimeError("boom"))
    builder = ContextBuilder(providers=[provider], timeout_ms=200)
    result = await builder.build(shell)
    assert result == []


async def test_builder_times_out_slow_provider(shell: MagicMock) -> None:
    async def slow(_: object) -> ContextFragment:
        await asyncio.sleep(10)
        return ContextFragment(provider="slow", summary="slow", payload={})

    provider = MagicMock()
    provider.collect = slow
    builder = ContextBuilder(providers=[provider], timeout_ms=50)
    result = await builder.build(shell)
    assert result == []
```

```python
# tests/context/test_providers.py
"""Tests for GitProvider and FilesystemProvider."""

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from agentsh.context.providers.git import GitProvider
from agentsh.context.providers.filesystem import FilesystemProvider
from agentsh.models import CommandResult


@pytest.fixture
def shell() -> MagicMock:
    return MagicMock()


async def test_git_provider_returns_fragment_in_git_repo(shell: MagicMock) -> None:
    shell.execute = AsyncMock(
        side_effect=[
            CommandResult(
                stdout="main\n", stderr="", exit_code=0, duration_ms=1, cwd="/repo"
            ),
            CommandResult(
                stdout=" M file.py\n",
                stderr="",
                exit_code=0,
                duration_ms=1,
                cwd="/repo",
            ),
        ]
    )
    provider = GitProvider()
    result = await provider.collect(shell)
    assert result is not None
    assert result.payload["branch"] == "main"


async def test_git_provider_returns_none_outside_repo(shell: MagicMock) -> None:
    shell.execute = AsyncMock(
        return_value=CommandResult(
            stdout="",
            stderr="fatal: not a git repository\n",
            exit_code=128,
            duration_ms=1,
            cwd="/tmp",
        )
    )
    provider = GitProvider()
    result = await provider.collect(shell)
    assert result is None


async def test_filesystem_provider_returns_fragment(
    shell: MagicMock, tmp_path: Path
) -> None:
    shell.cwd = AsyncMock(return_value=str(tmp_path))
    (tmp_path / "main.py").touch()
    provider = FilesystemProvider()
    result = await provider.collect(shell)
    assert result is not None
    assert "main.py" in result.payload.get("files", [])
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/context/ -v
```

- [ ] **Step 3: Implement context protocol and builder**

```python
# src/agentsh/context/__init__.py
"""Context collection layer — gathers environmental fragments for the agent."""

from agentsh.context.builder import ContextBuilder
from agentsh.context.protocol import ContextProvider

__all__ = ["ContextBuilder", "ContextProvider"]
```

```python
# src/agentsh/context/protocol.py
"""ContextProvider protocol definition."""

from __future__ import annotations

from typing import Protocol

from agentsh.models import ContextFragment
from agentsh.shell.protocol import Shell


class ContextProvider(Protocol):
    """Collects a single environmental ContextFragment from the shell."""

    name: str

    async def collect(self, shell: Shell) -> ContextFragment | None:
        """Return a fragment, or None if not applicable in the current environment."""
        ...
```

```python
# src/agentsh/context/builder.py
"""ContextBuilder — runs all providers concurrently with per-provider timeouts."""

from __future__ import annotations

import asyncio

from agentsh.context.protocol import ContextProvider
from agentsh.models import ContextFragment
from agentsh.shell.protocol import Shell


class ContextBuilder:
    """Collects context fragments from all configured providers in parallel."""

    def __init__(self, providers: list[ContextProvider], timeout_ms: int = 200) -> None:
        self._providers = providers
        self._timeout = timeout_ms / 1000

    async def build(self, shell: Shell) -> list[ContextFragment]:
        """Return all successfully collected fragments; failures are silently dropped."""

        async def _safe_collect(p: ContextProvider) -> ContextFragment | None:
            try:
                return await asyncio.wait_for(p.collect(shell), timeout=self._timeout)
            except Exception:
                return None

        results = await asyncio.gather(*(_safe_collect(p) for p in self._providers))
        return [r for r in results if r is not None]
```

- [ ] **Step 4: Implement GitProvider and FilesystemProvider**

```python
# src/agentsh/context/providers/__init__.py
"""Context provider implementations."""
```

```python
# src/agentsh/context/providers/git.py
"""Git context provider — reports current branch and working-tree status."""

from __future__ import annotations

from agentsh.models import ContextFragment
from agentsh.shell.protocol import Shell


class GitProvider:
    """Collects current git branch and dirty-file summary."""

    name = "git"

    async def collect(self, shell: Shell) -> ContextFragment | None:
        """Return git context, or None if not inside a git repository."""
        branch_result = await shell.execute(
            "git rev-parse --abbrev-ref HEAD 2>/dev/null"
        )
        if branch_result.exit_code != 0 or not branch_result.stdout.strip():
            return None

        status_result = await shell.execute("git status --short 2>/dev/null")
        changed_files = [
            line[3:].strip()
            for line in status_result.stdout.splitlines()
            if line.strip()
        ]

        return ContextFragment(
            provider=self.name,
            summary=f"git branch: {branch_result.stdout.strip()}",
            payload={
                "branch": branch_result.stdout.strip(),
                "changed_files": changed_files,
            },
        )
```

```python
# src/agentsh/context/providers/filesystem.py
"""Filesystem context provider — reports cwd contents."""

from __future__ import annotations

from pathlib import Path

from agentsh.models import ContextFragment
from agentsh.shell.protocol import Shell

_MAX_FILES = 50


class FilesystemProvider:
    """Collects a listing of the current working directory."""

    name = "filesystem"

    async def collect(self, shell: Shell) -> ContextFragment | None:
        """Return the top-level file listing of the current directory."""
        cwd = await shell.cwd()
        try:
            entries = sorted(Path(cwd).iterdir(), key=lambda p: (p.is_file(), p.name))
            files = [p.name + ("/" if p.is_dir() else "") for p in entries[:_MAX_FILES]]
        except OSError:
            return None

        return ContextFragment(
            provider=self.name,
            summary=f"cwd: {cwd} ({len(files)} entries)",
            payload={"cwd": cwd, "files": files},
        )
```

- [ ] **Step 5: Run tests**

```bash
uv run pytest tests/context/ -v
```

Expected: all pass.

- [ ] **Step 6: Lint and type-check**

```bash
uv run ruff check . && uv run ruff format . && uv run mypy src/
```

- [ ] **Step 7: Commit**

```bash
git add src/agentsh/context/ tests/context/
git commit -m "feat: add context providers and ContextBuilder with timeouts"
```

---

## Task 11: AnthropicAgent + AgentRouter

**Files:**

- Create: `src/agentsh/agent/__init__.py`
- Create: `src/agentsh/agent/protocol.py`
- Create: `src/agentsh/agent/router.py`
- Create: `src/agentsh/agent/anthropic.py`
- Create: `tests/agent/__init__.py`
- Create: `tests/agent/test_anthropic.py`

**Interfaces:**

- Consumes: `Message`, `ToolCall`, `ToolResult`, `ContextFragment` from `agentsh.models`; `AgentBackendConfig` from `agentsh.config`
- Produces: `Agent` protocol; `AgentRouter`; `AnthropicAgent` — used by the agentic loop in `run_repl`.

**Note:** The `ANTHROPIC_API_KEY` env var must be set at runtime. Tests use a recorded response fixture (no live calls).

- [ ] **Step 1: Write failing tests**

```python
# tests/agent/__init__.py
```

```python
# tests/agent/test_anthropic.py
"""Contract tests for AnthropicAgent using a mocked HTTP client."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from agentsh.agent.anthropic import AnthropicAgent
from agentsh.config import AgentBackendConfig
from agentsh.models import ContextFragment, Message, ToolCall


@pytest.fixture
def config() -> AgentBackendConfig:
    return AgentBackendConfig(model="claude-haiku-4-5-20251001", web_fetch=False)


@pytest.fixture
def text_response() -> MagicMock:
    block = MagicMock()
    block.type = "text"
    block.text = "Hello from the agent."
    response = MagicMock()
    response.content = [block]
    return response


@pytest.fixture
def tool_use_response() -> MagicMock:
    block = MagicMock()
    block.type = "tool_use"
    block.id = "tu_abc"
    block.name = "RunCommand"
    block.input = {"command": "ls -la"}
    response = MagicMock()
    response.content = [block]
    return response


async def test_respond_returns_text_message(
    config: AgentBackendConfig, text_response: MagicMock
) -> None:
    agent = AnthropicAgent(config)
    with patch.object(
        agent._client.messages, "create", new=AsyncMock(return_value=text_response)
    ):
        result = await agent.respond(
            conversation=[Message(role="user", content="hello")],
            context=[],
            tools=[],
        )
    assert result.role == "assistant"
    assert result.content == "Hello from the agent."
    assert result.tool_calls == ()


async def test_respond_parses_tool_calls(
    config: AgentBackendConfig, tool_use_response: MagicMock
) -> None:
    agent = AnthropicAgent(config)
    with patch.object(
        agent._client.messages, "create", new=AsyncMock(return_value=tool_use_response)
    ):
        result = await agent.respond(
            conversation=[Message(role="user", content="list files")],
            context=[],
            tools=[
                {
                    "name": "RunCommand",
                    "description": "run a command",
                    "input_schema": {},
                }
            ],
        )
    assert len(result.tool_calls) == 1
    tc = result.tool_calls[0]
    assert tc.tool_name == "RunCommand"
    assert tc.arguments == {"command": "ls -la"}
    assert tc.call_id == "tu_abc"
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/agent/ -v
```

- [ ] **Step 3: Implement Agent protocol and AgentRouter**

```python
# src/agentsh/agent/__init__.py
"""Agent layer — LLM backends and routing."""

from agentsh.agent.anthropic import AnthropicAgent
from agentsh.agent.protocol import Agent
from agentsh.agent.router import AgentRouter

__all__ = ["Agent", "AgentRouter", "AnthropicAgent"]
```

```python
# src/agentsh/agent/protocol.py
"""Agent protocol definition."""

from __future__ import annotations

from typing import Any, Protocol

from agentsh.models import ContextFragment, Message


class Agent(Protocol):
    """Interface for an LLM backend."""

    async def respond(
        self,
        conversation: list[Message],
        context: list[ContextFragment],
        tools: list[dict[str, Any]],
    ) -> Message:
        """Return the next assistant message given conversation history and context."""
        ...
```

```python
# src/agentsh/agent/router.py
"""AgentRouter — selects the active agent from the configured default."""

from __future__ import annotations

from agentsh.agent.protocol import Agent
from agentsh.config import AgentConfig


class AgentRouter:
    """Routes agent requests to the configured default backend."""

    def __init__(self, config: AgentConfig, agents: dict[str, Agent]) -> None:
        self._config = config
        self._agents = agents

    def current(self) -> Agent:
        """Return the active agent (currently always the configured default)."""
        return self._agents[self._config.default]
```

- [ ] **Step 4: Implement AnthropicAgent**

````python
# src/agentsh/agent/anthropic.py
"""Anthropic Claude backend."""

from __future__ import annotations

import json
from typing import Any

import anthropic

from agentsh.config import AgentBackendConfig
from agentsh.models import ContextFragment, Message, ToolCall

_SYSTEM_PREFIX = (
    "You are an AI assistant integrated into the user's shell. "
    "Use the provided tools to help with tasks. "
    "Be concise — you are running inside a terminal."
)


def _build_system(context: list[ContextFragment]) -> str:
    """Combine the base system prompt with serialized context fragments."""
    parts = [_SYSTEM_PREFIX]
    for frag in context:
        parts.append(
            f"\n## {frag.summary}\n```json\n{json.dumps(frag.payload, indent=2)}\n```"
        )
    return "\n".join(parts)


def _message_to_anthropic(m: Message) -> dict[str, Any]:
    """Convert a canonical Message to Anthropic's message format."""
    if m.tool_results:
        return {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tr.call_id,
                    "content": tr.content,
                    "is_error": tr.is_error,
                }
                for tr in m.tool_results
            ],
        }

    content: list[dict[str, Any]] = []
    if m.content:
        content.append({"type": "text", "text": m.content})
    for tc in m.tool_calls:
        content.append(
            {
                "type": "tool_use",
                "id": tc.call_id,
                "name": tc.tool_name,
                "input": tc.arguments,
            }
        )

    if len(content) == 1 and content[0]["type"] == "text":
        return {"role": m.role, "content": m.content}
    return {"role": m.role, "content": content}


class AnthropicAgent:
    """LLM backend using the Anthropic Messages API."""

    def __init__(self, config: AgentBackendConfig) -> None:
        self._config = config
        self._client = anthropic.AsyncAnthropic()

    async def respond(
        self,
        conversation: list[Message],
        context: list[ContextFragment],
        tools: list[dict[str, Any]],
    ) -> Message:
        """Call the Anthropic API and return the next assistant message."""
        anthropic_tools = [
            {
                "name": t["name"],
                "description": t["description"],
                "input_schema": t.get(
                    "input_schema", {"type": "object", "properties": {}}
                ),
            }
            for t in tools
        ]

        response = await self._client.messages.create(
            model=self._config.model,
            max_tokens=4096,
            system=_build_system(context),
            messages=[_message_to_anthropic(m) for m in conversation],
            tools=anthropic_tools,  # type: ignore[arg-type]
        )

        tool_calls = tuple(
            ToolCall(
                tool_name=block.name,
                arguments=dict(block.input),  # type: ignore[arg-type]
                call_id=block.id,
            )
            for block in response.content
            if block.type == "tool_use"
        )

        text_content = " ".join(
            block.text  # type: ignore[union-attr]
            for block in response.content
            if block.type == "text"
        )

        return Message(role="assistant", content=text_content, tool_calls=tool_calls)
````

- [ ] **Step 5: Run tests**

```bash
uv run pytest tests/agent/ -v
```

Expected: all pass (no live API calls).

- [ ] **Step 6: Lint and type-check**

```bash
uv run ruff check . && uv run ruff format . && uv run mypy src/
```

- [ ] **Step 7: Commit**

```bash
git add src/agentsh/agent/ tests/agent/
git commit -m "feat: add AnthropicAgent and AgentRouter"
```

---

## Task 12: Agentic Loop + Wire Agent into REPL

**Files:**

- Create: `src/agentsh/agent_loop.py`
- Modify: `src/agentsh/app.py`
- Modify: `src/agentsh/repl.py`
- Modify: `src/agentsh/main.py`

**Interfaces:**

- Consumes: All previously built components.
- Produces: End-to-end agent path: `classify → context → agent_loop → render`. The loop iterates until the agent produces a message with no tool calls.

**CONFIRM handling in agent loop:** When the permission engine returns CONFIRM for a tool call, `UI.confirm()` is shown. If the user denies, a `ToolResult(is_error=True, content="Permission denied by user.")` is added and the loop continues (the agent can decide what to do next).

- [ ] **Step 1: Implement the agentic loop**

```python
# src/agentsh/agent_loop.py
"""The agentic tool-call loop — runs until the agent stops requesting tools."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any

from agentsh.models import Message, ToolCall, ToolResult
from agentsh.permissions import PermissionLevel
from agentsh.tools.run_command import PermissionDeniedError

if TYPE_CHECKING:
    from agentsh.agent.protocol import Agent
    from agentsh.context.protocol import ContextProvider
    from agentsh.models import ContextFragment
    from agentsh.permissions import PermissionEngine
    from agentsh.repl import UI
    from agentsh.tools.protocol import ToolRegistry


def _tool_call_key(tc: ToolCall) -> str:
    """Build the permission key for a tool call."""
    match tc.tool_name:
        case "RunCommand":
            return f"RunCommand:{tc.arguments.get('command', '')}"
        case _:
            return tc.tool_name


async def run_agent_loop(
    *,
    agent: "Agent",
    conversation: list[Message],
    context: list["ContextFragment"],
    tools: "ToolRegistry",
    permissions: "PermissionEngine",
    ui: "UI",
) -> Message:
    """Run the agent until it produces a final response with no tool calls.

    Each tool call is evaluated by the permission engine; CONFIRM prompts the user;
    DENY injects an error tool result so the agent can recover gracefully.
    """
    while True:
        response = await agent.respond(conversation, context, tools.schemas())
        conversation.append(response)

        if not response.tool_calls:
            return response

        tool_results: list[ToolResult] = []
        for tc in response.tool_calls:
            key = _tool_call_key(tc)
            level = permissions.evaluate(key)

            match level:
                case PermissionLevel.DENY:
                    tool_results.append(
                        ToolResult(
                            call_id=tc.call_id,
                            content="Permission denied by policy.",
                            is_error=True,
                        )
                    )
                    continue
                case PermissionLevel.CONFIRM:
                    allowed = await ui.confirm(tc.tool_name, tc.arguments)
                    if not allowed:
                        tool_results.append(
                            ToolResult(
                                call_id=tc.call_id,
                                content="Permission denied by user.",
                                is_error=True,
                            )
                        )
                        continue

            try:
                tool = tools.get(tc.tool_name)
                result: Any = await tool.invoke(**tc.arguments)
                content = (
                    str(result)
                    if not hasattr(result, "stdout")
                    else (
                        f"stdout: {result.stdout}\nstderr: {result.stderr}\nexit_code: {result.exit_code}"
                    )
                )
                tool_results.append(ToolResult(call_id=tc.call_id, content=content))
            except PermissionDeniedError as e:
                tool_results.append(
                    ToolResult(call_id=tc.call_id, content=str(e), is_error=True)
                )
            except Exception as e:
                tool_results.append(
                    ToolResult(call_id=tc.call_id, content=f"Error: {e}", is_error=True)
                )

        conversation.append(
            Message(role="tool", content="", tool_results=tuple(tool_results))
        )
```

- [ ] **Step 2: Expand App to hold agent components**

```python
# src/agentsh/app.py
"""App — the top-level wiring object; holds all runtime dependencies."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from agentsh.models import Message
from agentsh.shell.protocol import Shell
from agentsh.tools.protocol import ToolRegistry

if TYPE_CHECKING:
    from agentsh.agent.router import AgentRouter
    from agentsh.context.builder import ContextBuilder
    from agentsh.permissions import PermissionEngine
    from agentsh.repl import UI


@dataclass
class AppState:
    """Mutable runtime state shared across REPL turns."""

    conversation: list[Message] = field(default_factory=list)


@dataclass
class App:
    """Dependency container; constructed in main.py and passed to run_repl."""

    shell: Shell
    tools: ToolRegistry
    permissions: "PermissionEngine"
    context_builder: "ContextBuilder"
    agent_router: "AgentRouter"
    state: AppState
    ui: "UI | None" = None
```

- [ ] **Step 3: Update run_repl to handle both paths**

Replace `run_repl` in `src/agentsh/repl.py`:

```python
async def run_repl(app: App) -> None:
    """Run the main REPL loop until EOF or KeyboardInterrupt."""
    from agentsh.agent_loop import run_agent_loop
    from agentsh.classifier import classify, InputKind
    from agentsh.models import Message
    from agentsh.permissions import PermissionLevel
    from agentsh.tools.run_command import PermissionDeniedError

    history_dir = Path.home() / ".local" / "share" / "agentsh"
    history_dir.mkdir(parents=True, exist_ok=True)
    session: PromptSession[str] = PromptSession(
        history=FileHistory(str(history_dir / "history"))
    )
    ui = UI(session)
    app.ui = ui

    while True:
        try:
            prompt = await app.shell.render_prompt()
            raw: str = await session.prompt_async(ANSI(prompt))
        except EOFError:
            break
        except KeyboardInterrupt:
            continue

        raw = raw.strip()
        if not raw:
            continue

        await app.shell.append_history(raw)
        kind = classify(raw, app.shell)

        match kind:
            case InputKind.SHELL_PARSEABLE:
                key = f"RunCommand:{raw}"
                level = app.permissions.evaluate(key)
                if level == PermissionLevel.DENY:
                    print(f"[agentsh] denied: {raw}", file=sys.stderr)
                    continue
                if level == PermissionLevel.CONFIRM:
                    if not await ui.confirm("RunCommand", {"command": raw}):
                        print("[agentsh] cancelled.", file=sys.stderr)
                        continue
                try:
                    result = await app.tools.get("RunCommand").invoke(command=raw)
                    ui.render(result)
                except PermissionDeniedError as e:
                    print(f"[agentsh] {e}", file=sys.stderr)

            case InputKind.AGENT:
                query = raw.removeprefix("/agent ").strip()
                context = await app.context_builder.build(app.shell)
                app.state.conversation.append(Message(role="user", content=query))
                final = await run_agent_loop(
                    agent=app.agent_router.current(),
                    conversation=app.state.conversation,
                    context=context,
                    tools=app.tools,
                    permissions=app.permissions,
                    ui=ui,
                )
                ui.render(final)
```

- [ ] **Step 4: Update main.py to wire everything**

```python
# src/agentsh/main.py
"""CLI entry point."""

from __future__ import annotations

import asyncio

from agentsh.agent.anthropic import AnthropicAgent
from agentsh.agent.router import AgentRouter
from agentsh.app import App, AppState
from agentsh.config import load_config
from agentsh.context.builder import ContextBuilder
from agentsh.context.providers.filesystem import FilesystemProvider
from agentsh.context.providers.git import GitProvider
from agentsh.permissions import PermissionEngine
from agentsh.repl import run_repl
from agentsh.shell.bash import BashShell
from agentsh.tools.protocol import ToolRegistry
from agentsh.tools.run_command import RunCommand


def _build_app() -> App:
    config = load_config()
    shell = BashShell()
    permissions = PermissionEngine(config.permissions)
    tools = ToolRegistry()
    tools.register(RunCommand(shell=shell, permissions=permissions))

    providers = [GitProvider(), FilesystemProvider()]
    context_builder = ContextBuilder(
        providers=providers, timeout_ms=config.context.timeout_ms
    )

    agents = {
        name: AnthropicAgent(backend_cfg)
        for name, backend_cfg in config.agent.backends.items()
    }
    agent_router = AgentRouter(config=config.agent, agents=agents)

    return App(
        shell=shell,
        tools=tools,
        permissions=permissions,
        context_builder=context_builder,
        agent_router=agent_router,
        state=AppState(),
    )


def main() -> None:
    """Entry point for the agentsh CLI."""
    app = _build_app()
    asyncio.run(run_repl(app))
```

- [ ] **Step 5: Smoke test the agent path**

```bash
ANTHROPIC_API_KEY=your_key uv run agentsh
```

Type `what files are in this directory` — the agent should respond using context from `FilesystemProvider`. Type `ls` — the shell path should execute directly.

- [ ] **Step 6: Run all tests**

```bash
uv run pytest -v
```

- [ ] **Step 7: Lint and type-check**

```bash
uv run ruff check . && uv run ruff format . && uv run mypy src/
```

- [ ] **Step 8: Commit**

```bash
git add src/agentsh/agent_loop.py src/agentsh/app.py src/agentsh/repl.py src/agentsh/main.py
git commit -m "feat: add agentic loop and wire classifier + agent into REPL"
```

---

## Task 13: ReadFile + WriteFile Tools

**Files:**

- Create: `src/agentsh/tools/read_file.py`
- Create: `src/agentsh/tools/write_file.py`
- Modify: `src/agentsh/tools/__init__.py`
- Modify: `src/agentsh/main.py`
- Create: `tests/tools/test_read_file.py`
- Create: `tests/tools/test_write_file.py`

**Interfaces:**

- Produces: `ReadFile` and `WriteFile` tools registered in `ToolRegistry`.

**WriteFile patch format:** A patch string may contain one or more SEARCH/REPLACE blocks:

```text
<<<<<<< SEARCH
old content
=======
new content
>>>>>>> REPLACE
```

Each block replaces the first occurrence of the SEARCH text. If `patch` is provided, `content` is ignored. If the search text is not found, `WriteFile.invoke` raises `ValueError`.

- [ ] **Step 1: Write failing tests**

```python
# tests/tools/test_read_file.py
"""Tests for ReadFile tool."""

import pytest
from pathlib import Path
from agentsh.tools.read_file import ReadFile


async def test_read_existing_file(tmp_path: Path) -> None:
    f = tmp_path / "hello.txt"
    f.write_text("hello world")
    tool = ReadFile()
    result = await tool.invoke(path=str(f))
    assert result == "hello world"


async def test_read_missing_file_raises(tmp_path: Path) -> None:
    tool = ReadFile()
    with pytest.raises(FileNotFoundError):
        await tool.invoke(path=str(tmp_path / "missing.txt"))
```

```python
# tests/tools/test_write_file.py
"""Tests for WriteFile tool."""

import pytest
from pathlib import Path
from agentsh.tools.write_file import WriteFile


async def test_full_write(tmp_path: Path) -> None:
    f = tmp_path / "out.txt"
    tool = WriteFile()
    await tool.invoke(path=str(f), content="new content")
    assert f.read_text() == "new content"


async def test_patch_replaces_block(tmp_path: Path) -> None:
    f = tmp_path / "code.py"
    f.write_text("def foo():\n    return 1\n")
    tool = WriteFile()
    patch = "<<<<<<< SEARCH\n    return 1\n=======\n    return 42\n>>>>>>> REPLACE"
    await tool.invoke(path=str(f), patch=patch)
    assert "return 42" in f.read_text()


async def test_patch_raises_if_search_not_found(tmp_path: Path) -> None:
    f = tmp_path / "code.py"
    f.write_text("def foo(): pass\n")
    tool = WriteFile()
    patch = "<<<<<<< SEARCH\nmissing\n=======\nreplaced\n>>>>>>> REPLACE"
    with pytest.raises(ValueError, match="not found"):
        await tool.invoke(path=str(f), patch=patch)


async def test_requires_content_or_patch(tmp_path: Path) -> None:
    tool = WriteFile()
    with pytest.raises(ValueError, match="content or patch"):
        await tool.invoke(path=str(tmp_path / "x.txt"))
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/tools/test_read_file.py tests/tools/test_write_file.py -v
```

- [ ] **Step 3: Implement ReadFile**

```python
# src/agentsh/tools/read_file.py
"""ReadFile tool — reads a file from the filesystem."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class ReadFile:
    """Reads a file and returns its contents as a string."""

    name = "ReadFile"
    description = "Read the contents of a file at the given path."
    schema: dict[str, Any] = {
        "name": "ReadFile",
        "description": "Read the contents of a file at the given path.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute or relative path to the file.",
                }
            },
            "required": ["path"],
        },
    }

    async def invoke(self, **kwargs: Any) -> str:
        """Return the file's contents; raises FileNotFoundError if absent."""
        path = Path(kwargs["path"])
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        return path.read_text(errors="replace")
```

- [ ] **Step 4: Implement WriteFile**

```python
# src/agentsh/tools/write_file.py
"""WriteFile tool — writes or patches a file on the filesystem."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

_BLOCK_RE = re.compile(
    r"<<<<<<< SEARCH\n(.*?)\n=======\n(.*?)\n>>>>>>> REPLACE",
    re.DOTALL,
)


def _apply_patch(original: str, patch: str) -> str:
    """Apply SEARCH/REPLACE blocks from patch to original, in order."""
    result = original
    blocks = _BLOCK_RE.findall(patch)
    if not blocks:
        raise ValueError("Patch contains no valid SEARCH/REPLACE blocks.")
    for search, replacement in blocks:
        if search not in result:
            raise ValueError(f"Search text not found in file: {search[:80]!r}")
        result = result.replace(search, replacement, 1)
    return result


class WriteFile:
    """Writes content to a file, or applies a SEARCH/REPLACE patch."""

    name = "WriteFile"
    description = (
        "Write content to a file (full overwrite), or apply targeted SEARCH/REPLACE edits "
        "using the patch parameter."
    )
    schema: dict[str, Any] = {
        "name": "WriteFile",
        "description": (
            "Write content to a file (full overwrite), or apply targeted SEARCH/REPLACE edits "
            "using the patch parameter."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute or relative path to the file.",
                },
                "content": {
                    "type": "string",
                    "description": "Full file content for a complete overwrite. Mutually exclusive with patch.",
                },
                "patch": {
                    "type": "string",
                    "description": (
                        "One or more SEARCH/REPLACE blocks. Format: "
                        "<<<<<<< SEARCH\\n<old>\\n=======\\n<new>\\n>>>>>>> REPLACE"
                    ),
                },
            },
            "required": ["path"],
        },
    }

    async def invoke(self, **kwargs: Any) -> str:
        """Write or patch the file; returns a confirmation string."""
        path = Path(kwargs["path"])
        content: str | None = kwargs.get("content")
        patch: str | None = kwargs.get("patch")

        if patch is None and content is None:
            raise ValueError("WriteFile requires either content or patch.")

        path.parent.mkdir(parents=True, exist_ok=True)

        if patch is not None:
            original = path.read_text(errors="replace") if path.exists() else ""
            path.write_text(_apply_patch(original, patch))
        else:
            path.write_text(content or "")

        return f"Written: {path}"
```

- [ ] **Step 5: Register tools in main.py**

In `src/agentsh/main.py`, inside `_build_app`, add after `RunCommand`:

```python
from agentsh.tools.read_file import ReadFile
from agentsh.tools.write_file import WriteFile

tools.register(ReadFile())
tools.register(WriteFile())
```

- [ ] **Step 6: Run all tests**

```bash
uv run pytest -v
```

- [ ] **Step 7: Lint and type-check**

```bash
uv run ruff check . && uv run ruff format . && uv run mypy src/
```

- [ ] **Step 8: Commit**

```bash
git add src/agentsh/tools/ tests/tools/ src/agentsh/main.py
git commit -m "feat: add ReadFile and WriteFile tools with patch mode"
```

---

## Task 14: Remaining Context Providers

**Files:**

- Create: `src/agentsh/context/providers/python_env.py`
- Create: `src/agentsh/context/providers/docker.py`
- Create: `src/agentsh/context/providers/kubernetes.py`
- Create: `src/agentsh/context/providers/history.py`
- Create: `src/agentsh/context/providers/environment.py`
- Modify: `src/agentsh/main.py`
- Modify: `tests/context/test_providers.py`

**Interfaces:**

- Produces: 5 additional `ContextProvider` implementations. Each returns `None` silently if the relevant tool/env is absent.

- [ ] **Step 1: Add provider tests**

Append to `tests/context/test_providers.py`:

```python
from agentsh.context.providers.python_env import PythonEnvProvider
from agentsh.context.providers.docker import DockerProvider
from agentsh.context.providers.history import HistoryProvider
from agentsh.context.providers.environment import EnvironmentProvider


async def test_python_env_provider(shell: MagicMock) -> None:
    shell.execute = AsyncMock(
        return_value=CommandResult(
            stdout="Python 3.12.0\n", stderr="", exit_code=0, duration_ms=1, cwd="/"
        )
    )
    shell.cwd = AsyncMock(return_value="/project")
    provider = PythonEnvProvider()
    result = await provider.collect(shell)
    assert result is not None
    assert "3.12" in result.payload.get("python_version", "")


async def test_docker_provider_returns_none_without_docker(shell: MagicMock) -> None:
    shell.execute = AsyncMock(
        return_value=CommandResult(
            stdout="",
            stderr="docker: command not found\n",
            exit_code=127,
            duration_ms=1,
            cwd="/",
        )
    )
    provider = DockerProvider()
    result = await provider.collect(shell)
    assert result is None


async def test_history_provider(shell: MagicMock) -> None:
    shell.history = AsyncMock(return_value=["ls -la", "cd /tmp", "git status"])
    provider = HistoryProvider()
    result = await provider.collect(shell)
    assert result is not None
    assert "git status" in result.payload.get("recent", [])


async def test_environment_provider(shell: MagicMock) -> None:
    shell.env = AsyncMock(
        return_value={"HOME": "/home/user", "PATH": "/usr/bin", "SECRET_KEY": "abc123"}
    )
    provider = EnvironmentProvider()
    result = await provider.collect(shell)
    assert result is not None
    assert "SECRET_KEY" not in result.payload.get("env", {})
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/context/test_providers.py -v
```

- [ ] **Step 3: Implement PythonEnvProvider**

```python
# src/agentsh/context/providers/python_env.py
"""Python environment context provider."""

from __future__ import annotations

from pathlib import Path

from agentsh.models import ContextFragment
from agentsh.shell.protocol import Shell


class PythonEnvProvider:
    """Reports the Python version and presence of venv/pyproject.toml."""

    name = "python"

    async def collect(self, shell: Shell) -> ContextFragment | None:
        """Return Python environment info, or None if Python is not available."""
        version_result = await shell.execute(
            "python3 --version 2>/dev/null || python --version 2>/dev/null"
        )
        if version_result.exit_code != 0 or not version_result.stdout.strip():
            return None

        cwd = await shell.cwd()
        has_venv = (Path(cwd) / ".venv").exists() or (Path(cwd) / "venv").exists()
        has_pyproject = (Path(cwd) / "pyproject.toml").exists()

        return ContextFragment(
            provider=self.name,
            summary=f"python: {version_result.stdout.strip()}",
            payload={
                "python_version": version_result.stdout.strip(),
                "has_venv": has_venv,
                "has_pyproject": has_pyproject,
            },
        )
```

- [ ] **Step 4: Implement DockerProvider**

```python
# src/agentsh/context/providers/docker.py
"""Docker context provider — reports running containers."""

from __future__ import annotations

from agentsh.models import ContextFragment
from agentsh.shell.protocol import Shell


class DockerProvider:
    """Reports running Docker containers if Docker is available."""

    name = "docker"

    async def collect(self, shell: Shell) -> ContextFragment | None:
        """Return container info, or None if Docker is unavailable."""
        result = await shell.execute(
            "docker ps --format '{{.Names}}\t{{.Image}}\t{{.Status}}' 2>/dev/null"
        )
        if result.exit_code != 0 or not result.stdout.strip():
            return None

        containers = [
            dict(zip(["name", "image", "status"], line.split("\t")))
            for line in result.stdout.splitlines()
            if "\t" in line
        ]

        return ContextFragment(
            provider=self.name,
            summary=f"docker: {len(containers)} container(s) running",
            payload={"containers": containers},
        )
```

- [ ] **Step 5: Implement KubernetesProvider**

```python
# src/agentsh/context/providers/kubernetes.py
"""Kubernetes context provider — reports current kubectl context."""

from __future__ import annotations

from agentsh.models import ContextFragment
from agentsh.shell.protocol import Shell


class KubernetesProvider:
    """Reports the active kubectl context if kubectl is configured."""

    name = "kubernetes"

    async def collect(self, shell: Shell) -> ContextFragment | None:
        """Return the active k8s context, or None if kubectl is unavailable."""
        result = await shell.execute("kubectl config current-context 2>/dev/null")
        if result.exit_code != 0 or not result.stdout.strip():
            return None

        return ContextFragment(
            provider=self.name,
            summary=f"kubernetes context: {result.stdout.strip()}",
            payload={"context": result.stdout.strip()},
        )
```

- [ ] **Step 6: Implement HistoryProvider**

```python
# src/agentsh/context/providers/history.py
"""History context provider — surfaces recent shell commands."""

from __future__ import annotations

from agentsh.models import ContextFragment
from agentsh.shell.protocol import Shell

_HISTORY_LIMIT = 20


class HistoryProvider:
    """Provides the last N shell commands as context."""

    name = "history"

    async def collect(self, shell: Shell) -> ContextFragment | None:
        """Return recent command history."""
        recent = await shell.history(limit=_HISTORY_LIMIT)
        if not recent:
            return None

        return ContextFragment(
            provider=self.name,
            summary=f"recent commands ({len(recent)})",
            payload={"recent": recent},
        )
```

- [ ] **Step 7: Implement EnvironmentProvider**

```python
# src/agentsh/context/providers/environment.py
"""Environment context provider — surfaces safe env vars."""

from __future__ import annotations

from agentsh.models import ContextFragment
from agentsh.shell.protocol import Shell

_SAFE_KEYS = frozenset(
    {
        "USER",
        "HOME",
        "SHELL",
        "TERM",
        "LANG",
        "PWD",
        "EDITOR",
        "VIRTUAL_ENV",
        "CONDA_DEFAULT_ENV",
        "GOPATH",
        "GOROOT",
        "NODE_ENV",
        "RAILS_ENV",
        "RACK_ENV",
    }
)

_SECRET_SUBSTRINGS = (
    "KEY",
    "SECRET",
    "TOKEN",
    "PASSWORD",
    "PASS",
    "CREDENTIAL",
    "AUTH",
)


def _is_safe(key: str) -> bool:
    """Return True if the env key is unlikely to contain sensitive data."""
    if key in _SAFE_KEYS:
        return True
    upper = key.upper()
    return not any(sub in upper for sub in _SECRET_SUBSTRINGS)


class EnvironmentProvider:
    """Provides a filtered subset of env vars, excluding anything that looks like a secret."""

    name = "environment"

    async def collect(self, shell: Shell) -> ContextFragment | None:
        """Return safe environment variables."""
        env = await shell.env()
        safe = {k: v for k, v in env.items() if _is_safe(k)}

        return ContextFragment(
            provider=self.name,
            summary=f"environment ({len(safe)} safe vars)",
            payload={"env": safe},
        )
```

- [ ] **Step 8: Update main.py to include all providers**

In `_build_app`, replace the providers list:

```python
from agentsh.context.providers.docker import DockerProvider
from agentsh.context.providers.environment import EnvironmentProvider
from agentsh.context.providers.filesystem import FilesystemProvider
from agentsh.context.providers.git import GitProvider
from agentsh.context.providers.history import HistoryProvider
from agentsh.context.providers.kubernetes import KubernetesProvider
from agentsh.context.providers.python_env import PythonEnvProvider

_ALL_PROVIDERS = {
    "git": GitProvider,
    "filesystem": FilesystemProvider,
    "python": PythonEnvProvider,
    "docker": DockerProvider,
    "kubernetes": KubernetesProvider,
    "history": HistoryProvider,
    "environment": EnvironmentProvider,
}

providers = [
    _ALL_PROVIDERS[name]()
    for name in config.context.providers
    if name in _ALL_PROVIDERS
]
```

- [ ] **Step 9: Run all tests**

```bash
uv run pytest -v
```

- [ ] **Step 10: Lint and type-check**

```bash
uv run ruff check . && uv run ruff format . && uv run mypy src/
```

- [ ] **Step 11: Commit**

```bash
git add src/agentsh/context/ tests/context/ src/agentsh/main.py
git commit -m "feat: add remaining context providers"
```

---

## Task 15: EventBus

**Files:**

- Create: `src/agentsh/events.py`
- Modify: `src/agentsh/app.py`
- Modify: `src/agentsh/repl.py`
- Modify: `src/agentsh/agent_loop.py`
- Modify: `src/agentsh/main.py`
- Create: `tests/test_events.py`

**Interfaces:**

- Produces: `EventBus`; `CommandStarted`, `CommandFinished`, `ToolInvoked`, `ToolDenied`, `AgentResponded`, `ContextCollected` event dataclasses. Published at the relevant call sites; subscribers registered at startup.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_events.py
"""Tests for EventBus publish/subscribe ordering and subscriber isolation."""

import pytest
from agentsh.events import CommandFinished, EventBus


async def test_subscriber_receives_event() -> None:
    bus = EventBus()
    received: list[CommandFinished] = []
    bus.subscribe(CommandFinished, received.append)
    event = CommandFinished(command="ls", exit_code=0, duration_ms=1.0)
    await bus.publish(event)
    assert received == [event]


async def test_multiple_subscribers_all_called() -> None:
    bus = EventBus()
    log: list[str] = []
    bus.subscribe(CommandFinished, lambda e: log.append("first"))
    bus.subscribe(CommandFinished, lambda e: log.append("second"))
    await bus.publish(CommandFinished(command="pwd", exit_code=0, duration_ms=0.5))
    assert log == ["first", "second"]


async def test_subscriber_exception_does_not_stop_others() -> None:
    bus = EventBus()
    log: list[str] = []

    async def bad(e: object) -> None:
        raise RuntimeError("boom")

    bus.subscribe(CommandFinished, bad)
    bus.subscribe(CommandFinished, lambda e: log.append("ok"))
    await bus.publish(CommandFinished(command="echo", exit_code=0, duration_ms=0.1))
    assert log == ["ok"]


def test_unrelated_event_not_delivered() -> None:
    bus = EventBus()
    received: list[object] = []
    bus.subscribe(CommandFinished, received.append)
    # ToolDenied is a different type — should not trigger CommandFinished subscribers
    from agentsh.events import ToolDenied
    import asyncio

    asyncio.run(
        bus.publish(ToolDenied(tool_name="RunCommand", key="RunCommand:rm -rf /"))
    )
    assert received == []
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/test_events.py -v
```

- [ ] **Step 3: Implement EventBus**

```python
# src/agentsh/events.py
"""EventBus and core event types for cross-cutting observability."""

from __future__ import annotations

import asyncio
import inspect
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable


class EventBus:
    """Simple async event bus; subscriber exceptions are swallowed to preserve isolation."""

    def __init__(self) -> None:
        self._subscribers: dict[type, list[Callable[..., Any]]] = defaultdict(list)

    def subscribe(self, event_type: type, handler: Callable[..., Any]) -> None:
        """Register handler to be called for every published event of event_type."""
        self._subscribers[event_type].append(handler)

    async def publish(self, event: Any) -> None:
        """Deliver event to all registered subscribers; swallow individual handler errors."""
        for handler in self._subscribers[type(event)]:
            try:
                result = handler(event)
                if inspect.isawaitable(result):
                    await result
            except Exception:
                pass


@dataclass(frozen=True)
class CommandStarted:
    """Published immediately before a shell command is sent to the backend."""

    command: str
    cwd: str = ""


@dataclass(frozen=True)
class CommandFinished:
    """Published after a shell command returns."""

    command: str
    exit_code: int
    duration_ms: float


@dataclass(frozen=True)
class ToolInvoked:
    """Published after a tool call completes (success or error)."""

    tool_name: str
    arguments: dict[str, Any]
    success: bool


@dataclass(frozen=True)
class ToolDenied:
    """Published when a tool call is blocked by the permission engine."""

    tool_name: str
    key: str


@dataclass(frozen=True)
class AgentResponded:
    """Published each time the agent returns a message in the agentic loop."""

    content: str
    tool_call_count: int


@dataclass(frozen=True)
class ContextCollected:
    """Published after ContextBuilder finishes collecting all fragments."""

    provider_count: int
    fragment_count: int
```

- [ ] **Step 4: Add EventBus to App and wire publish calls**

In `src/agentsh/app.py`, add `event_bus: EventBus` field.

In `src/agentsh/repl.py` shell path, publish `CommandStarted` before and `CommandFinished` after `RunCommand.invoke`.

In `src/agentsh/agent_loop.py`, publish `ToolInvoked`, `ToolDenied`, and `AgentResponded` at the relevant call sites.

In `src/agentsh/context/builder.py`, publish `ContextCollected` after gathering.

In `src/agentsh/main.py`, create `EventBus()` and pass it into `App`.

- [ ] **Step 5: Run all tests**

```bash
uv run pytest -v
```

- [ ] **Step 6: Lint and type-check**

```bash
uv run ruff check . && uv run ruff format . && uv run mypy src/
```

- [ ] **Step 7: Commit**

```bash
git add src/agentsh/events.py tests/test_events.py src/agentsh/
git commit -m "feat: add EventBus and core events"
```

---

## Self-Review

**Spec coverage:**

| Spec section                                  | Task(s)            |
| --------------------------------------------- | ------------------ |
| §5 Core models                                | Task 2             |
| §6 Shell layer / BashShell                    | Task 3             |
| §7 Classifier                                 | Task 9             |
| §8 Context providers                          | Tasks 10, 14       |
| §9 Agent layer / AnthropicAgent               | Task 11            |
| §10 Tools / RunCommand / ReadFile / WriteFile | Tasks 4, 13        |
| §11 Permission engine                         | Tasks 7, 8         |
| §12 Event bus                                 | Task 15            |
| §13 Config schema                             | Task 6             |
| §14 REPL loop                                 | Tasks 5, 8, 12     |
| §15 Testing strategy                          | Tests in each task |
| §16 Phase 1–4                                 | Tasks 1–15         |
| AgentRouter (§9 partial)                      | Task 11            |
| Agentic tool-use loop (implicit §14)          | Task 12            |

**Out of scope for this plan (Phase 5 per spec §16):** PowerShell/Cmd backends; OpenAIAgent, OpenRouterAgent, LocalAgent; plugin hooks via event bus.

**Type consistency check:** `ToolCall.call_id` defined in Task 2 → used as `tc.call_id` in Tasks 11, 12 ✓. `ContextFragment.payload` is `dict[str, Any]` throughout ✓. `PermissionEngine.evaluate` returns `PermissionLevel` in Task 7 → matched in Tasks 8, 12 ✓. `ToolRegistry.schemas()` returns `list[dict[str, Any]]` ✓.
