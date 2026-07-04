"""Git context provider — reports current branch and working-tree status."""

from __future__ import annotations

from agentsh.models import ContextFragment
from agentsh.shell.protocol import Shell


class GitProvider:
    """Collects current git branch and dirty-file summary."""

    name = "git"

    async def collect(self, shell: Shell) -> ContextFragment | None:
        """Return git context, or None if not inside a git repository."""
        branch_result = await shell.execute(
            "git rev-parse --abbrev-ref HEAD 2>/dev/null"
        )
        if branch_result.exit_code != 0 or not branch_result.stdout.strip():
            return None

        status_result = await shell.execute("git status --short 2>/dev/null")
        changed_files = [
            line[3:].strip()
            for line in status_result.stdout.splitlines()
            if line.strip()
        ]

        return ContextFragment(
            provider=self.name,
            summary=f"git branch: {branch_result.stdout.strip()}",
            payload={
                "branch": branch_result.stdout.strip(),
                "changed_files": changed_files,
            },
        )
