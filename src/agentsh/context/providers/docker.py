"""Docker context provider — reports running containers."""

from agentsh.models import ContextFragment
from agentsh.shell.protocol import Shell


class DockerProvider:
    """Collects a list of running Docker containers."""

    name = "docker"

    async def collect(self, shell: Shell) -> ContextFragment | None:
        """Return running containers, or None if Docker is unavailable.

        No stderr redirection is used here: ``CommandResult`` already
        separates stdout/stderr/exit_code regardless of what the command
        does with fd 2, and POSIX-only redirection syntax such as
        ``2>/dev/null`` breaks on cmd.exe and PowerShell. The ``--format``
        value is double-quoted rather than single-quoted for the same
        reason: cmd.exe does not treat single quotes as a quoting
        character, so they would be passed through literally.
        """
        result = await shell.execute(
            'docker ps --format "{{.Names}}\t{{.Image}}\t{{.Status}}"'
        )
        if result.exit_code != 0:
            return None

        containers = [
            dict(
                zip(["name", "image", "status"], line.split("\t"), strict=False)
            )
            for line in result.stdout.splitlines()
            if line.strip()
        ]

        return ContextFragment(
            provider=self.name,
            summary=f"{len(containers)} running container(s)",
            payload={"containers": containers},
        )
