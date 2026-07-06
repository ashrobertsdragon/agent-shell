"""Input classifier — decides whether to route input to shell or agent."""

from enum import Enum, auto

from agentsh.shell.protocol import Shell


class InputKind(Enum):
    """The result of classifying a raw user input string."""

    AGENT = auto()
    SHELL = auto()


async def classify(raw: str, shell: Shell) -> InputKind:
    """Return AGENT or SHELL based on input prefix and shell parse check."""
    if raw == "/agent" or raw.startswith("/agent "):
        return InputKind.AGENT
    if await shell.can_parse(raw):
        return InputKind.SHELL
    return InputKind.AGENT


def agent_query(raw: str) -> str:
    """Return the agent query with any /agent prefix removed.

    Only a whole-word prefix is stripped, so free-form input that
    merely starts with the characters "/agent" is left intact.
    """
    if raw == "/agent" or raw.startswith("/agent "):
        return raw.removeprefix("/agent").strip()
    return raw.strip()
