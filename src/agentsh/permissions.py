"""Permission engine — evaluates tool calls against allow/confirm/deny rules."""

import shlex
from collections.abc import Awaitable, Callable, Mapping
from enum import Enum, auto
from fnmatch import fnmatch

from agentsh.config import PermissionRulesConfig
from agentsh.models import JsonValue

_RUN_COMMAND_PREFIX = "RunCommand:"

_SHELL_METACHARACTERS = frozenset(";&|$`<>(){}\n\r\\%!\x00")


class PermissionLevel(Enum):
    """Outcome of a permission evaluation."""

    ALLOW = auto()
    CONFIRM = auto()
    DENY = auto()


def _command_has_shell_metacharacters(command: str) -> bool:
    r"""Return True if a shell would interpret part of command specially.

    Covers chaining/substitution operators (``;``, ``&``, ``|``, ``$``,
    backticks), redirection (``<``, ``>``), subshells/grouping
    (``(``, ``)``, ``{``, ``}``), escaping (``\\``), embedded newlines,
    cmd.exe variable expansion (``%``, ``!``), and null bytes. Also
    treats commands that ``shlex`` cannot tokenize (e.g. unbalanced
    quotes) as suspicious, since their real behavior under a shell is
    ambiguous.
    """
    if not _SHELL_METACHARACTERS.isdisjoint(command):
        return True
    try:
        shlex.split(command)
    except ValueError:
        return True
    return False


class PermissionDeniedError(Exception):
    """Raised when a tool call is blocked by policy or lacks confirmation."""


type ConfirmCallback = Callable[[str, Mapping[str, JsonValue]], Awaitable[bool]]


def tool_call_key(tool_name: str, arguments: Mapping[str, JsonValue]) -> str:
    """Build the canonical permission key for a tool call.

    This is the single source of truth for key construction; every call
    site (the agent loop, the REPL, and each tool's own self-enforcement)
    must use it so alternate spellings of the same call cannot evaluate
    against different keys.

    The key format is:
      - ``"RunCommand:{command}"`` with the command stripped
      - ``"ReadFile:{path}"`` / ``"WriteFile:{path}"`` with the path
        resolved to an absolute POSIX-style form
      - ``"{tool_name}"`` for any other tool
    """
    from agentsh.tools._paths import canonical_path

    match tool_name:
        case "RunCommand":
            command = str(arguments.get("command", "")).strip()
            return f"RunCommand:{command}"
        case "ReadFile" | "WriteFile":
            path = canonical_path(str(arguments.get("path", "")))
            return f"{tool_name}:{path.as_posix()}"
        case _:
            return tool_name


class PermissionEngine:
    """Evaluates a tool_call_key against declarative fnmatch rules.

    Deny is checked first so a broad deny rule cannot be overridden by a
    narrower allow or confirm rule.

    For ``RunCommand`` keys, any shell metacharacter in the command
    (``;``, ``&``, ``|``, ``$``, backticks, redirection, etc.) forces at
    least CONFIRM: an allow-rule glob such as ``"RunCommand:git *"`` can
    never fnmatch its way past a chained or substituted command, since
    fnmatch's ``*`` spans those characters just like any other.

    The tool_call_key format is:
      - ``"RunCommand:{command}"`` with the command stripped
      - ``"ReadFile:{path}"`` / ``"WriteFile:{path}"`` with the path
        resolved to an absolute POSIX-style form
      - ``"{tool_name}"`` for any other tool
    """

    def __init__(self, rules: PermissionRulesConfig) -> None:
        """Initialise with a set of allow/confirm/deny glob patterns."""
        self._rules = rules

    def evaluate(self, tool_call_key: str) -> PermissionLevel:
        """Return the permission level for the given tool call key."""
        if any(fnmatch(tool_call_key, p) for p in self._rules.deny):
            return PermissionLevel.DENY
        if any(fnmatch(tool_call_key, p) for p in self._rules.confirm):
            return PermissionLevel.CONFIRM

        if tool_call_key.startswith(_RUN_COMMAND_PREFIX):
            command = tool_call_key.removeprefix(_RUN_COMMAND_PREFIX)
            if _command_has_shell_metacharacters(command):
                return PermissionLevel.CONFIRM

        if any(fnmatch(tool_call_key, p) for p in self._rules.allow):
            return PermissionLevel.ALLOW
        return PermissionLevel.CONFIRM

    async def enforce(
        self,
        tool_name: str,
        arguments: Mapping[str, JsonValue],
        confirm: ConfirmCallback | None,
    ) -> None:
        """Enforce policy for a tool call, regardless of who is calling.

        DENY always raises PermissionDeniedError. CONFIRM raises unless a
        confirm callback is supplied and it approves the call. ALLOW
        returns without side effects. This is the mechanism every tool
        must call from its own invoke() so permission enforcement cannot
        be bypassed by calling the tool directly.
        """
        key = tool_call_key(tool_name, arguments)
        match self.evaluate(key):
            case PermissionLevel.DENY:
                raise PermissionDeniedError(
                    f"{tool_name} denied by policy: {key}"
                )
            case PermissionLevel.CONFIRM:
                if confirm is None:
                    raise PermissionDeniedError(
                        f"{tool_name} requires confirmation but no confirm"
                        f" callback is configured: {key}"
                    )
                if not await confirm(tool_name, arguments):
                    raise PermissionDeniedError(
                        f"{tool_name} denied by user: {key}"
                    )
            case PermissionLevel.ALLOW:
                pass
