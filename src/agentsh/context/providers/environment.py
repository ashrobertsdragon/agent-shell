"""Environment context provider — reports safe environment variables."""

from agentsh.context.providers import register
from agentsh.models import ContextFragment
from agentsh.shell.protocol import Shell

_SAFE_ALLOWLIST = {
    "HOME",
    "USER",
    "SHELL",
    "TERM",
    "TERM_PROGRAM",
    "COLORTERM",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "LC_MESSAGES",
    "PATH",
    "PWD",
    "OLDPWD",
    "EDITOR",
    "VISUAL",
    "TZ",
    "HOSTNAME",
    "DISPLAY",
    "XDG_SESSION_TYPE",
    "NODE_ENV",
    "VIRTUAL_ENV",
    "CONDA_DEFAULT_ENV",
}


@register("environment")
class EnvironmentProvider:
    """Collects non-sensitive environment variables from the shell."""

    name = "environment"

    async def collect(self, shell: Shell) -> ContextFragment | None:
        """Return only known-safe environment variables."""
        raw_env = await shell.env()
        safe_env = {k: v for k, v in raw_env.items() if k in _SAFE_ALLOWLIST}
        if not safe_env:
            return None

        return ContextFragment(
            provider=self.name,
            summary=f"{len(safe_env)} environment variables",
            payload={"env": safe_env},
        )
