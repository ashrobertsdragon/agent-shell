"""Kubernetes context provider — reports current context and namespace."""

from agentsh.models import ContextFragment
from agentsh.shell.protocol import Shell


class KubernetesProvider:
    """Collects the active kubectl context and namespace."""

    name = "kubernetes"

    async def collect(self, shell: Shell) -> ContextFragment | None:
        """Return the current kube context or None if kubectl is unavailable.

        No stderr redirection is used here: ``CommandResult`` already
        separates stdout/stderr/exit_code regardless of what the command
        does with fd 2, and POSIX-only redirection syntax such as
        ``2>/dev/null`` breaks on cmd.exe and PowerShell.
        """
        ctx_result = await shell.execute("kubectl config current-context")
        if ctx_result.exit_code != 0 or not ctx_result.stdout.strip():
            return None

        ns_result = await shell.execute(
            'kubectl config view --minify -o jsonpath="{..namespace}"'
        )
        namespace = ns_result.stdout.strip() or "default"

        return ContextFragment(
            provider=self.name,
            summary=f"k8s context: {ctx_result.stdout.strip()}",
            payload={
                "context": ctx_result.stdout.strip(),
                "namespace": namespace,
            },
        )
