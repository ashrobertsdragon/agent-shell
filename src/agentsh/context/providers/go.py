"""Go context provider — reports the installed Go version and module info."""

from collections.abc import Iterator
from pathlib import Path

from agentsh.context.providers import register
from agentsh.models import ContextFragment
from agentsh.shell.protocol import Shell


def _strip_comment(line: str) -> str:
    """Strip a trailing ``//`` comment and surrounding whitespace from a line.

    go.mod and go.work use ``//`` for trailing comments (frequently
    ``// indirect`` on ``require`` lines), never for path separators, so a
    naive split on the first occurrence is safe.
    """
    return line.split("//", 1)[0].strip()


def _parse_requirements(lines: Iterator[str]) -> list[dict[str, str]]:
    """Parse the body of a ``require (...)`` block up to its closing ``)``.

    Consumes lines from the shared iterator so the caller's outer loop
    resumes right after the block.
    """
    dependencies: list[dict[str, str]] = []
    for raw_line in lines:
        line = _strip_comment(raw_line)
        if line == ")":
            break
        parts = line.split()
        if len(parts) >= 2:
            dependencies.append({"path": parts[0], "version": parts[1]})
    return dependencies


def _parse_go_mod(
    text: str,
) -> tuple[str | None, str | None, list[dict[str, str]]]:
    """Parse a go.mod file's module path, go directive, and dependencies.

    go.mod is a custom line-oriented format, not TOML/JSON: each directive
    normally occupies its own line, but multi-value directives such as
    ``require`` may instead use a parenthesized block spanning several
    lines. Both forms are handled here.
    """
    module: str | None = None
    go_directive: str | None = None
    dependencies: list[dict[str, str]] = []

    lines = iter(text.splitlines())
    for raw_line in lines:
        line = _strip_comment(raw_line)
        if not line:
            continue
        match line.split(maxsplit=1):
            case ["module", rest]:
                module = rest.strip()
            case ["go", rest]:
                go_directive = rest.strip()
            case ["require", "("]:
                dependencies.extend(_parse_requirements(lines))
            case ["require", rest]:
                parts = rest.split()
                if len(parts) >= 2:
                    dependencies.append({"path": parts[0], "version": parts[1]})
    return module, go_directive, dependencies


def _parse_go_work(text: str) -> list[str]:
    """Parse a go.work file's ``use`` directives into member module dirs.

    Mirrors go.mod's shape: a single-line ``use <dir>`` directive or a
    parenthesized ``use (...)`` block listing one directory per line.
    """
    modules: list[str] = []
    lines = iter(text.splitlines())
    for raw_line in lines:
        line = _strip_comment(raw_line)
        if not line:
            continue
        match line.split(maxsplit=1):
            case ["use", "("]:
                for block_raw_line in lines:
                    block_line = _strip_comment(block_raw_line)
                    if block_line == ")":
                        break
                    if block_line:
                        modules.append(block_line)
            case ["use", rest]:
                modules.append(rest.strip())
    return modules


@register("go")
class GoProvider:
    """Collects the installed Go version and go.mod/go.work project info."""

    name = "go"

    async def collect(self, shell: Shell) -> ContextFragment | None:
        """Return Go version and module/workspace info.

        Returns None only when there is nothing worth reporting: no
        ``go`` binary on PATH and no go.mod in the current directory. A
        fragment is still returned when go.mod is absent but ``go`` is
        installed (the installed version is useful context on its own,
        e.g. outside a Go project directory), and likewise when go.mod
        is present but ``go`` itself is not installed.

        No stderr redirection is used here: ``CommandResult`` already
        separates stdout/stderr/exit_code regardless of what the command
        does with fd 2, and POSIX-only redirection syntax such as
        ``2>/dev/null`` breaks on cmd.exe and PowerShell.

        go.mod and go.work are read directly off the filesystem via
        ``Path(shell.cwd)`` rather than through a shell command, the same
        pattern PythonProvider uses for its ``.venv`` check.
        """
        version_result = await shell.execute("go version")
        go_version: str | None = None
        if version_result.exit_code == 0:
            tokens = version_result.stdout.split()
            if len(tokens) >= 3 and tokens[2].startswith("go"):
                go_version = tokens[2].removeprefix("go")

        go_mod_path = Path(shell.cwd) / "go.mod"
        module: str | None = None
        go_directive: str | None = None
        dependencies: list[dict[str, str]] = []
        if go_mod_path.is_file():
            try:
                go_mod_text = go_mod_path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                go_mod_text = None
            if go_mod_text is not None:
                module, go_directive, dependencies = _parse_go_mod(go_mod_text)

        go_work_path = Path(shell.cwd) / "go.work"
        workspace_modules: list[str] | None = None
        if go_work_path.is_file():
            try:
                go_work_text = go_work_path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                go_work_text = None
            if go_work_text is not None:
                workspace_modules = _parse_go_work(go_work_text)

        if go_version is None and module is None:
            return None

        summary_parts = [
            part
            for part in (
                f"go {go_version}" if go_version else None,
                f"module {module}" if module else None,
                f"workspace ({len(workspace_modules)} modules)"
                if workspace_modules
                else None,
            )
            if part
        ]

        return ContextFragment(
            provider=self.name,
            summary=", ".join(summary_parts) if summary_parts else "go",
            payload={
                "go_version": go_version,
                "module": module,
                "go_directive": go_directive,
                "dependencies": dependencies,
                "workspace_modules": workspace_modules,
            },
        )
