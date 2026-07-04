"""Docker context provider — reports running containers."""

from __future__ import annotations

from agentsh.models import ContextFragment
from agentsh.shell.protocol import Shell


class DockerProvider:
    """Collects a list of running Docker containers."""

    name = "docker"

    async def collect(self, shell: Shell) -> ContextFragment | None:
        """Return running containers, or None if Docker is unavailable."""
        result = await shell.execute(
            "docker ps --format '{{.Names}}\t{{.Image}}\t{{.Status}}' 2>/dev/null"
        )
        if result.exit_code != 0:
            return None

        containers = [
            dict(zip(["name", "image", "status"], line.split("\t"), strict=False))
            for line in result.stdout.splitlines()
            if line.strip()
        ]

        return ContextFragment(
            provider=self.name,
            summary=f"{len(containers)} running container(s)",
            payload={"containers": containers},
        )
