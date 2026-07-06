"""Shell protocol definition."""

from typing import Protocol, runtime_checkable

from agentsh.models import CommandResult


@runtime_checkable
class Shell(Protocol):
    """Interface for a persistent shell backend."""

    async def execute(self, command: str) -> CommandResult:
        """Execute a command and return its result."""
        ...

    @property
    def cwd(self) -> str:
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

    async def can_parse(self, raw: str) -> bool:
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
