"""Input classifier — decides whether to route input to shell or agent."""

from __future__ import annotations

from enum import Enum, auto

from agentsh.shell.protocol import Shell


class InputKind(Enum):
    """The result of classifying a raw user input string."""

    AGENT = auto()
    SHELL_PARSEABLE = auto()


def classify(raw: str, shell: Shell) -> InputKind:
    """Return AGENT or SHELL_PARSEABLE based on input prefix and shell parse check."""
    match raw:
        case s if s.startswith("/agent "):
            return InputKind.AGENT
        case s if shell.can_parse(s):
            return InputKind.SHELL_PARSEABLE
        case _:
            return InputKind.AGENT
