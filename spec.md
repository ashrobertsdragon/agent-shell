# agentsh — Technical Specification (v2)

## 1. Overview

`agentsh` is a shell wrapper that routes user input to either a native shell backend (Bash, PowerShell, Cmd) or an LLM agent, depending on intent. The agent has tool access to the shell and filesystem, gated by a permission engine. Python is the reference implementation; the design is meant to port cleanly to Rust or Go once proven.

## 2. Goals

- Single REPL that transparently handles both shell commands and natural-language requests.
- Shell execution is never special-cased. It's a tool like any other.
- Agent backend is swappable (Anthropic, OpenAI, OpenRouter, local).
- Permission policy is declarative and decoupled from tool implementation.
- Everything observable through an event bus.

## 3. Non-Goals

- No built-in shell language parsing or reimplementation.
- No plugin marketplace in v1.
- No multi-user or remote session support in v1.

## 4. Architecture Layers

```text
REPL
  → Input Dispatcher
    → Classifier
      → Shell Path    → Shell Backend → RunCommand Tool
      → Agent Path    → Context Builder → Agent Router → Tool Registry → Permission Engine
    → Response Renderer
  → Event Bus (cross-cuts everything)
```

## 5. Core Data Models

```python
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Protocol


class ExitCode(Enum):
    SUCCESS = 0
    FAILURE = 1
    DENIED = 126
    NOT_FOUND = 127


@dataclass(frozen=True, slots=True)
class CommandResult:
    stdout: str
    stderr: str
    exit_code: int
    duration_ms: float
    cwd: str


@dataclass(frozen=True, slots=True)
class Message:
    role: str
    content: str
    tool_calls: tuple["ToolCall", ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class ToolCall:
    tool_name: str
    arguments: dict[str, Any]
    call_id: str


@dataclass(frozen=True, slots=True)
class ContextFragment:
    provider: str
    summary: str
    payload: dict[str, Any]
```

## 6. Shell Layer

```python
class Shell(Protocol):
    async def execute(self, command: str) -> CommandResult: ...
    async def cwd(self) -> str: ...
    async def env(self) -> dict[str, str]: ...
    async def history(self, limit: int = 100) -> list[str]: ...
    async def complete(self, partial: str) -> list[str]: ...
    def can_parse(self, raw: str) -> bool: ...
    async def render_prompt(self) -> str: ...
    async def append_history(self, command: str) -> None: ...
```

Implementations: `BashShell`, `PowerShellShell`, `CmdShell`. Each wraps a persistent subprocess with a PTY or pipe, tracks cwd/env internally, and translates its own error semantics into `CommandResult`. No implementation parses or rewrites the command string; it's passed through verbatim.

Shell selection happens once at startup, from config or `$SHELL` detection, and is injected into `RunCommand`.

## 7. Classifier

```python
class InputKind(Enum):
    AGENT = auto()
    SHELL_PARSEABLE = auto()


def classify(raw: str, shell: Shell) -> InputKind:
    match raw:
        case s if s.startswith("/agent "):
            return InputKind.AGENT
        case s if shell.can_parse(s):
            return InputKind.SHELL_PARSEABLE
        case _:
            return InputKind.AGENT
```

## 8. Context Providers

```python
class ContextProvider(Protocol):
    name: str

    async def collect(self, shell: Shell) -> ContextFragment | None: ...
```

Providers: `GitProvider`, `FilesystemProvider`, `PythonProvider`, `DockerProvider`, `KubernetesProvider`, `HistoryProvider`, `EnvironmentProvider`.

```python
class ContextBuilder:
    def __init__(self, providers: list[ContextProvider]) -> None:
        self._providers = providers

    async def build(self, shell: Shell) -> list[ContextFragment]:
        results = await asyncio.gather(
            *(p.collect(shell) for p in self._providers),
            return_exceptions=True,
        )
        return [r for r in results if isinstance(r, ContextFragment)]
```

Each provider is independently timeboxed and failures are swallowed, not raised.

## 9. Agent Layer

```python
@dataclass(frozen=True, slots=True)
class AgentConfig:
    model: str
    web_fetch: bool = False


class Agent(Protocol):
    config: AgentConfig

    async def respond(
        self,
        conversation: list[Message],
        context: list[ContextFragment],
        tools: list["Tool"],
    ) -> Message: ...
```

Implementations: `AnthropicAgent`, `OpenAIAgent`, `OpenRouterAgent`, `LocalAgent`. Each checks `config.web_fetch` and, if supported and enabled, registers its own native web-fetch capability with the provider rather than exposing a local tool for it. `LocalAgent` ignores the flag.

```toml
[agent.anthropic]
model = "claude-sonnet-5"
web_fetch = true

[agent.openai]
model = "gpt-5"
web_fetch = true

[agent.local]
model = "llama-3"
web_fetch = false
```

**Note:** native web fetch executes on the provider's infrastructure and bypasses the permission engine entirely. This is a documented exception, not an oversight. Leave `web_fetch = false` for any agent where that's unacceptable and route fetch needs through `RunCommand:curl*` instead.

## 10. Tools

```python
class Tool(Protocol):
    name: str
    description: str
    schema: dict[str, Any]

    async def invoke(self, **kwargs: Any) -> Any: ...
```

v1 tool set, final:

- `RunCommand`
- `ReadFile`
- `WriteFile`

Everything that used to be a dedicated wrapper around a shell invocation (`GitStatus`, `GitCommit`, `DockerLogs`, `DockerPs`, `KillProcess`, `SearchFiles`, `OpenEditor`) collapses into `RunCommand`, permission-scoped by matching on the command string. `ReadFile`/`WriteFile` stay separate because they take structured arguments (path, content, optional patch mode), which lets permissions scope by path glob and lets the agent make precise edits instead of generating `sed` calls.

```python
class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        return self._tools[name]

    def schemas(self) -> list[dict[str, Any]]:
        return [t.schema for t in self._tools.values()]
```

## 11. Permission Engine

```python
class PermissionLevel(Enum):
    ALLOW = auto()
    CONFIRM = auto()
    DENY = auto()


@dataclass(frozen=True, slots=True)
class PermissionRules:
    allow: tuple[str, ...] = ()
    confirm: tuple[str, ...] = ()
    deny: tuple[str, ...] = ()


class PermissionEngine:
    def __init__(self, rules: PermissionRules) -> None:
        self._rules = rules

    def evaluate(self, tool_call_key: str) -> PermissionLevel:
        if any(fnmatch(tool_call_key, p) for p in self._rules.deny):
            return PermissionLevel.DENY
        if any(fnmatch(tool_call_key, p) for p in self._rules.confirm):
            return PermissionLevel.CONFIRM
        if any(fnmatch(tool_call_key, p) for p in self._rules.allow):
            return PermissionLevel.ALLOW
        return PermissionLevel.CONFIRM
```

`tool_call_key` is `"{tool_name}:{command}"` for `RunCommand`, `"{tool_name}"` for `ReadFile`/`WriteFile` unless path-scoping is added later. Deny is checked first, so a broad deny pattern can never be overridden by a narrower allow.

```toml
[permissions.rules]
allow = ["RunCommand:ls*", "RunCommand:pwd", "RunCommand:cat*", "RunCommand:git status", "ReadFile"]
confirm = ["RunCommand:sudo*", "RunCommand:git commit*", "RunCommand:git push*", "RunCommand:curl*", "RunCommand:wget*", "RunCommand:winget*", "RunCommand:irm*", "WriteFile"]
deny = ["RunCommand:rm -rf*"]
```

Default with no matching rule is `CONFIRM`, not `ALLOW`.

## 12. Event Bus

```python
class EventBus:
    def __init__(self) -> None:
        self._subscribers: dict[type, list[Callable]] = defaultdict(list)

    def subscribe(self, event_type: type, handler: Callable) -> None:
        self._subscribers[event_type].append(handler)

    async def publish(self, event: Any) -> None:
        for handler in self._subscribers[type(event)]:
            await handler(event)
```

Core events: `CommandStarted`, `CommandFinished`, `ToolInvoked`, `ToolDenied`, `AgentResponded`, `ContextCollected`.

## 13. Config Schema

```toml
[shell]
backend = "auto"

[agent]
default = "anthropic"

[agent.anthropic]
model = "claude-sonnet-5"
web_fetch = true

[context]
timeout_ms = 200
providers = ["git", "filesystem", "python", "docker"]

[permissions.rules]
allow = []
confirm = []
deny = []
```

## 14. REPL Loop

```python
async def run_repl(app: App) -> None:
    while True:
        raw = await app.ui.read_input()
        kind = classify(raw, app.shell)
        match kind:
            case InputKind.SHELL_PARSEABLE:
                result = await app.tools.get("RunCommand").invoke(command=raw)
            case InputKind.AGENT:
                context = await app.context_builder.build(app.shell)
                result = await app.agent_router.current().respond(
                    app.state.conversation, context, app.tools.schemas()
                )
        await app.ui.render(result)
```

## 15. Testing Strategy

- Shell backends: golden-file tests against a fake PTY, one suite per backend.
- Tools: unit tests with mocked shell/filesystem, plus permission-engine tests covering deny-precedence explicitly.
- Classifier: table-driven tests, one row per `InputKind`.
- Agent adapters: contract tests against recorded API responses, no live calls in CI.
- Event bus: assert publish order and subscriber isolation.

## 16. Phased Build Plan

1. Shell abstraction + BashShell + RunCommand tool + REPL loop with no agent.
1. Tool registry + permission engine, wired to RunCommand only.
1. Single agent adapter (Anthropic) + classifier + context builder with GitProvider and FilesystemProvider only.
1. ReadFile/WriteFile, remaining context providers, event bus, history.
1. PowerShell/Cmd backends, remaining agent adapters, plugin hooks via event bus.

## 17. Port-to-Rust/Go Considerations

`Shell`, `Agent`, `Tool`, `ContextProvider` map directly to traits or interfaces. The event bus maps to channels. `PermissionEngine.evaluate` is a straight port, three vector scans with a fixed check order. The classifier is trivial by design now, three-way match, nothing to prototype further before committing to a port.
