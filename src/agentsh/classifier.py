"""Input classifier — decides whether to route input to shell or agent."""

from enum import Enum, auto

from agentsh.shell.protocol import Shell


class InputKind(Enum):
    """The result of classifying a raw user input string."""

    AGENT = auto()
    SHELL = auto()


async def classify(raw: str, shell: Shell) -> InputKind:
    """Return AGENT or SHELL based on input prefix and shell parse check."""
    if raw.startswith("/agent "):
        return InputKind.AGENT
    if await shell.can_parse(raw):
        return InputKind.SHELL
    return InputKind.AGENT
