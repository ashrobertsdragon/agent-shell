"""Permission engine — evaluates tool calls against allow/confirm/deny rules."""

import shlex
from enum import Enum, auto
from fnmatch import fnmatch

from agentsh.config import PermissionRulesConfig

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
