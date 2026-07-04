"""Environment context provider — reports safe environment variables."""

from __future__ import annotations

from agentsh.models import ContextFragment
from agentsh.shell.protocol import Shell

_SENSITIVE_SUBSTRINGS = (
    "key",
    "secret",
    "token",
    "password",
    "passwd",
    "credential",
    "auth",
    "api",
)

_SAFE_ALLOWLIST = {"HOME", "USER", "SHELL", "TERM", "LANG", "PATH", "PWD", "OLDPWD"}


def _is_sensitive(name: str) -> bool:
    """Return True if the env var name looks like it might hold a secret."""
    lower = name.lower()
    return any(sub in lower for sub in _SENSITIVE_SUBSTRINGS)


class EnvironmentProvider:
    """Collects non-sensitive environment variables from the shell."""

    name = "environment"

    async def collect(self, shell: Shell) -> ContextFragment | None:
        """Return filtered env vars, excluding anything that looks sensitive."""
        raw_env = await shell.env()
        safe_env = {
            k: v
            for k, v in raw_env.items()
            if k in _SAFE_ALLOWLIST or not _is_sensitive(k)
        }
        if not safe_env:
            return None

        return ContextFragment(
            provider=self.name,
            summary=f"{len(safe_env)} environment variables",
            payload={"env": safe_env},
        )
