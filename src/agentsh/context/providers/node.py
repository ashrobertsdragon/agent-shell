"""Node.js context provider."""

import json
from collections.abc import Mapping
from pathlib import Path

from agentsh.context.providers import register
from agentsh.models import ContextFragment, JsonValue
from agentsh.shell.protocol import Shell


def _string_map(value: JsonValue | None) -> dict[str, str]:
    """Coerce a JSON value to a flat string-to-string mapping, or {}.

    ``package.json``'s ``scripts``, ``dependencies``, and
    ``devDependencies`` fields are always objects of string values in
    well-formed manifests; any other shape (missing key, malformed
    file) degrades to an empty mapping rather than raising.
    """
    if not isinstance(value, Mapping):
        return {}
    return {str(key): str(item) for key, item in value.items()}


@register("node")
class NodeProvider:
    """Collects Node.js version and package.json scripts/dependencies."""

    name = "node"

    async def collect(self, shell: Shell) -> ContextFragment | None:
        """Return Node version and package.json info, or None if Node is absent.

        No stderr redirection is used here: ``CommandResult`` already
        separates stdout/stderr/exit_code regardless of what the command
        does with fd 2, and POSIX-only redirection syntax such as
        ``2>/dev/null`` breaks on cmd.exe and PowerShell.

        ``package.json`` is read directly from disk via ``Path(shell.cwd)``
        rather than through a shell command, mirroring how
        ``PythonProvider`` checks for a ``.venv`` directory. A missing or
        malformed ``package.json`` does not prevent the fragment from
        being returned: the Node version alone is useful context even
        without a JS project in the current directory, so a missing file
        and malformed JSON both degrade to empty scripts/dependencies
        rather than returning ``None`` or raising.
        """
        version_result = await shell.execute("node --version")
        version_text = (
            version_result.stdout.strip() or version_result.stderr.strip()
        )
        if version_result.exit_code != 0 or not version_text:
            return None

        node_version = version_text.removeprefix("v")

        package_data: JsonValue = None
        package_json_path = Path(shell.cwd) / "package.json"
        if package_json_path.is_file():
            try:
                package_data = json.loads(package_json_path.read_text())
            except json.JSONDecodeError:
                package_data = None

        fields = package_data if isinstance(package_data, Mapping) else {}
        scripts = _string_map(fields.get("scripts"))
        dependencies = _string_map(fields.get("dependencies"))
        dev_dependencies = _string_map(fields.get("devDependencies"))

        return ContextFragment(
            provider=self.name,
            summary=f"node {node_version}",
            payload={
                "node_version": node_version,
                "scripts": scripts,
                "dependencies": dependencies,
                "dev_dependencies": dev_dependencies,
            },
        )
