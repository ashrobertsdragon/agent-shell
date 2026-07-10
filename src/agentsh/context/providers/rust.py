"""Rust context provider — reports toolchain and Cargo.toml dependencies."""

import asyncio
import tomllib
from pathlib import Path

from agentsh.context.providers import register
from agentsh.models import ContextFragment
from agentsh.shell.protocol import Shell


def _read_cargo_toml(path: Path) -> dict:
    """Parse a Cargo.toml manifest, run off the event loop."""
    with path.open("rb") as f:
        return tomllib.load(f)


def _dependency_version(spec: object) -> str:
    """Return a displayable version string for a dependency's value.

    Cargo.toml dependency values are either a bare version string
    (``serde = "1.0"``) or a table (``serde = { version = "1.0",
    features = [...] }``); workspace-inherited dependencies
    (``serde = { workspace = true }``) carry no version at all.
    """
    match spec:
        case str():
            return spec
        case dict() as table:
            version = table.get("version")
            if isinstance(version, str):
                return version
            if table.get("workspace") is True:
                return "workspace"
            return "*"
        case _:
            return "*"


@register("rust")
class RustProvider:
    """Collects the active Rust toolchain and Cargo.toml dependencies."""

    name = "rust"

    async def collect(self, shell: Shell) -> ContextFragment | None:
        """Return Rust toolchain and dependency info, or None if absent.

        No stderr redirection is used here: ``CommandResult`` already
        separates stdout/stderr/exit_code regardless of what the command
        does with fd 2, and POSIX-only redirection syntax such as
        ``2>/dev/null`` breaks on cmd.exe and PowerShell.

        ``rustc`` and ``cargo`` are typically installed together via
        rustup, but either can be missing (e.g. a system package only
        installs one), so both are probed and either is enough to
        establish that a Rust toolchain is present. ``rustup show
        active-toolchain`` gives a named toolchain (e.g.
        ``stable-x86_64-unknown-linux-gnu``) when rustup itself is
        installed; when it isn't (rustc/cargo from a system package
        manager, no rustup), the toolchain description falls back to
        the raw ``rustc --version`` (or ``cargo --version``) string.

        Cargo.toml is read directly from the filesystem rather than via
        a shell command, mirroring how python.py checks for a local
        .venv. A missing or unparseable Cargo.toml does not make the
        provider return None: the toolchain is useful context on its
        own even outside a cargo project directory, so only package
        name/version/dependencies are left empty in that case.
        """
        rustc_result = await shell.execute("rustc --version")
        rustc_version = (
            rustc_result.stdout.strip() or rustc_result.stderr.strip()
        )
        cargo_result = await shell.execute("cargo --version")
        cargo_version = (
            cargo_result.stdout.strip() or cargo_result.stderr.strip()
        )

        have_rustc = rustc_result.exit_code == 0 and bool(rustc_version)
        have_cargo = cargo_result.exit_code == 0 and bool(cargo_version)
        if not have_rustc and not have_cargo:
            return None

        toolchain = rustc_version if have_rustc else cargo_version

        rustup_result = await shell.execute("rustup show active-toolchain")
        rustup_text = (
            rustup_result.stdout.strip() or rustup_result.stderr.strip()
        )
        if rustup_result.exit_code == 0 and rustup_text:
            toolchain = rustup_text.splitlines()[0].strip()

        package_name: str | None = None
        package_version: str | None = None
        dependencies: dict[str, str] = {}

        cargo_toml_path = Path(shell.cwd) / "Cargo.toml"
        if cargo_toml_path.is_file():
            try:
                manifest = await asyncio.to_thread(
                    _read_cargo_toml, cargo_toml_path
                )
            except (OSError, tomllib.TOMLDecodeError):
                manifest = None

            if manifest is not None:
                package = manifest.get("package")
                if isinstance(package, dict):
                    name = package.get("name")
                    package_name = name if isinstance(name, str) else None
                    version = package.get("version")
                    package_version = (
                        version if isinstance(version, str) else None
                    )

                deps = manifest.get("dependencies")
                if isinstance(deps, dict):
                    dependencies = {
                        dep_name: _dependency_version(spec)
                        for dep_name, spec in deps.items()
                    }

        summary_parts = [f"rust {toolchain}"]
        if package_name:
            package_label = package_name
            if package_version:
                package_label += f" {package_version}"
            summary_parts.append(package_label)
        if dependencies:
            summary_parts.append(f"{len(dependencies)} dep(s)")

        return ContextFragment(
            provider=self.name,
            summary=", ".join(summary_parts),
            payload={
                "toolchain": toolchain,
                "package_name": package_name,
                "package_version": package_version,
                "dependencies": dependencies,
            },
        )
