"""Permission engine — evaluates tool calls against allow/confirm/deny rules."""

from enum import Enum, auto
from fnmatch import fnmatch

from agentsh.config import PermissionRulesConfig


class PermissionLevel(Enum):
    """Outcome of a permission evaluation."""

    ALLOW = auto()
    CONFIRM = auto()
    DENY = auto()


class PermissionEngine:
    """Evaluates a tool_call_key against declarative fnmatch rules.

    Deny is checked first so a broad deny rule cannot be overridden by a
    narrower allow or confirm rule.

    The tool_call_key format is:
      - ``"{tool_name}:{command}"`` for RunCommand
      - ``"{tool_name}"`` for ReadFile / WriteFile
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
        if any(fnmatch(tool_call_key, p) for p in self._rules.allow):
            return PermissionLevel.ALLOW
        return PermissionLevel.CONFIRM
