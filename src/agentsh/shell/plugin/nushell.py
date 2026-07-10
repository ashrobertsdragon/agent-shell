"""Persistent-looking Nushell shell backend backed by one-shot processes.

Nushell is structured-data, not POSIX text streams, and -- critically --
refuses to run as a REPL when its stdin is not a TTY ("Nushell launched
as a REPL, but STDIN is not a TTY"). That rules out the sentinel-over-a
long-lived-pipe protocol every other backend in this package uses (see
``_base.ProcessBackedShell``): there is no way to keep one ``nu`` process
alive across calls and feed it commands one at a time the way bash/cmd/
PowerShell are fed.

Instead, NuShellShell spawns a fresh, non-interactive ``nu -c <script>``
process for every ``execute()`` call (Nushell's documented
"command-string" execution mode, which does not require a TTY). State
that must survive between calls -- only the working directory, here --
is threaded through explicitly: each call starts ``nu`` with
``cwd=self._cwd`` and reads the resulting directory back out of a
sentinel line, exactly like the persistent backends do. Environment
mutations made via ``$env.FOO = ...`` do NOT persist across calls, since
there is no live process to carry them forward; this is a deliberate,
documented limitation rather than an oversight (see NuShellShell's
docstring).

The sentinel line reuses the same ``marker:exit_code:cwd`` convention
and the same ``read_until_sentinel`` stream reader as bash/cmd/
PowerShell, which also gives it the same output-capping behaviour
(MAX_OUTPUT_BYTES) for free. What differs is only how the wrapper script
itself is built: a plain ``try`` block sets ``$env.LAST_EXIT_CODE`` even
for pure-internal (non-external) commands, since Nushell only updates
that variable itself after running an external process, and a caught
error is reported as exit code 1 with its message written to stderr.
"""

import asyncio
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from agentsh.history_security import append_secure_line
from agentsh.limits import read_capped_text, read_last_lines
from agentsh.models import CommandResult
from agentsh.shell._registry import register
from agentsh.shell.plugin._base import new_marker
from agentsh.shell.plugin._stderr_file import (
    create_stderr_tempfile,
    discard_stderr_tempfile,
)
from agentsh.shell.plugin._stream import read_until_sentinel

_SENTINEL = "__AGENTSH_NU_DONE_8675309__"

_CHECK_TIMEOUT = 2.0


def _default_history_path() -> Path:
    """Return agentsh's own nushell history file, beside its config."""
    return Path.home() / ".config" / "agentsh" / "nushell_history"


def _parse_sentinel(line: str, marker: str) -> tuple[int, str] | None:
    """Return (exit_code, cwd) if line is an exact sentinel match for marker.

    maxsplit=2 keeps cwd intact as the final field even if it contains a
    colon. The first field must equal marker exactly (not merely start
    with it), so command output that happens to contain sentinel-like
    text cannot be mistaken for the real one; marker itself carries a
    per-call random nonce (see new_marker) so it cannot be forged.
    """
    parts = line.rstrip("\r\n").split(":", 2)
    if len(parts) != 3 or parts[0] != marker:
        return None
    try:
        return int(parts[1]), parts[2]
    except ValueError:
        return None


def _wrap_command(command: str, marker: str) -> str:
    """Wrap command so a caught error and the final cwd are always reported.

    $env.LAST_EXIT_CODE is reset to 0 first: Nushell only assigns it
    itself after running an external command, so a purely-internal
    command (e.g. ``1 + 1``) would otherwise leave it unset. The
    surrounding try/catch guarantees the sentinel line is always printed
    -- even when command fails -- so the resulting cwd is never lost;
    without it, an uncaught error would abort the script before the
    print statement below ever ran.
    """
    return (
        "$env.LAST_EXIT_CODE = 0\n"
        "try {\n"
        f"{command}\n"
        "} catch {|err|\n"
        "    print --stderr $err.msg\n"
        "    $env.LAST_EXIT_CODE = 1\n"
        "}\n"
        f'print $"{marker}:($env.LAST_EXIT_CODE):($env.PWD)"\n'
    )


@register("nu")
class NuShellShell:
    """Wraps Nushell as a sequence of one-shot ``nu -c`` subprocesses.

    Registered as ``"nu"``, not ``"nushell"``: per CONTRIBUTING.md, the
    registered name must be what ``detect_shell()`` returns for
    auto-detection to find it, and for POSIX shells that's the basename
    of ``$SHELL`` (see ``_detect.py``) -- which is ``nu`` for a Nushell
    login shell, not ``nushell``. Registering under any other string
    would silently break ``shell = "auto"`` for Nushell users without
    ``_detect.py`` changes, exactly as the fish/zsh backends rely on
    their registered name matching their executable's basename.

    Does not subclass ProcessBackedShell: that base class's whole
    purpose is managing one long-lived subprocess's lifecycle (lazy
    start, restart on exit/desync, forced reset), and Nushell has no
    long-lived subprocess to manage here -- see the module docstring for
    why a persistent, pipe-fed ``nu`` process isn't viable. `reset` and
    `close` still do something real (killing an in-flight one-shot
    process on cancellation) but there is no "restart on next use"
    concept to inherit.

    Only cwd is carried across calls. Environment variable assignments
    (``$env.FOO = ...``) do not persist between execute() calls, since
    each call is a fresh process; the alternative (serialising and
    replaying the whole ``$env`` on every call) was judged too fragile
    to get right without a local Nushell install to verify against, so
    this limitation is documented rather than silently half-implemented.
    """

    def __init__(self) -> None:
        """Initialise cwd tracking, the command lock, and lazy exe lookup."""
        self._cwd = os.getcwd()
        self._exe: str | None = None
        self._lock = asyncio.Lock()
        self._current_proc: asyncio.subprocess.Process | None = None
        self._history_path = _default_history_path()

    @property
    def cwd(self) -> str:
        """Return the last tracked working directory."""
        return self._cwd

    def _resolve_exe(self) -> str:
        """Return the nu executable path, resolving and caching it lazily.

        Raises:
            RuntimeError: If no nu executable is found on PATH.
        """
        if self._exe is None:
            exe = shutil.which("nu")
            if exe is None:
                raise RuntimeError("no nu executable found")
            self._exe = exe
        return self._exe

    async def _spawn(
        self, exe: str, script: str, stderr_path: str
    ) -> asyncio.subprocess.Process:
        """Start a one-shot nu subprocess with stderr redirected to a file.

        Unlike the persistent backends (which toggle stderr redirection
        mid-script because many commands share one long-lived process),
        this process runs exactly one command, so its stderr fd can be
        fixed for the whole process at spawn time.
        """
        stderr_file = await asyncio.to_thread(open, stderr_path, "wb")
        try:
            return await asyncio.create_subprocess_exec(
                exe,
                "--no-config-file",
                "-c",
                script,
                cwd=self._cwd,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=stderr_file,
            )
        finally:
            await asyncio.to_thread(stderr_file.close)

    async def execute(self, command: str) -> CommandResult:
        """Execute a Nushell command in a fresh, non-interactive nu process.

        Output is capped at MAX_OUTPUT_BYTES via the shared
        read_until_sentinel helper, exactly as for the persistent
        backends. If the sentinel line never arrives (e.g. the command
        called ``exit`` directly, bypassing the wrapper's try/catch),
        the process's own exit code is used and cwd is left unchanged,
        but any output collected before that point is still returned.

        Args:
            command (str): The Nushell command or script to run.

        Returns:
            CommandResult: The command stdout, stderr, exit code, and cwd.
        """
        async with self._lock:
            exe = self._resolve_exe()
            stderr_path = await asyncio.to_thread(create_stderr_tempfile)
            start = time.monotonic()
            marker = new_marker(_SENTINEL)
            script = _wrap_command(command, marker)

            try:
                try:
                    proc = await self._spawn(exe, script, stderr_path)
                except OSError as e:
                    duration_ms = (time.monotonic() - start) * 1000
                    return CommandResult(
                        stdout="",
                        stderr=str(e),
                        exit_code=1,
                        duration_ms=duration_ms,
                        cwd=self._cwd,
                    )

                self._current_proc = proc
                try:
                    if proc.stdout is None:
                        stdout_content, sentinel_line = "", ""
                    else:
                        (
                            stdout_content,
                            sentinel_line,
                        ) = await read_until_sentinel(proc.stdout, f"{marker}:")
                    await proc.wait()
                finally:
                    self._current_proc = None

                parsed = (
                    _parse_sentinel(sentinel_line, marker)
                    if sentinel_line
                    else None
                )
                stderr_content = await asyncio.to_thread(
                    read_capped_text, stderr_path
                )
                duration_ms = (time.monotonic() - start) * 1000

                if parsed is None:
                    return CommandResult(
                        stdout=stdout_content,
                        stderr=stderr_content,
                        exit_code=proc.returncode or 1,
                        duration_ms=duration_ms,
                        cwd=self._cwd,
                    )

                exit_code, self._cwd = parsed
                return CommandResult(
                    stdout=stdout_content,
                    stderr=stderr_content,
                    exit_code=exit_code,
                    duration_ms=duration_ms,
                    cwd=self._cwd,
                )
            finally:
                await asyncio.to_thread(discard_stderr_tempfile, stderr_path)

    async def env(self) -> dict[str, str]:
        """Return the tracked environment by running `$env | to json -r`.

        Only string-valued entries are kept: nushell's $env can hold
        non-string values (e.g. closures like PROMPT_COMMAND) that
        cannot round-trip as process environment variables anyway.
        """
        result = await self.execute("$env | to json -r")
        try:
            raw = json.loads(result.stdout)
        except (json.JSONDecodeError, TypeError):
            return {}
        if not isinstance(raw, dict):
            return {}
        return {k: v for k, v in raw.items() if isinstance(v, str)}

    async def history(self, limit: int = 100) -> list[str]:
        """Return the last `limit` lines of agentsh's own nu history file."""
        try:
            return await asyncio.to_thread(
                read_last_lines, self._history_path, limit
            )
        except FileNotFoundError:
            return []

    async def complete(self, partial: str) -> list[str]:
        """Return up to 20 PATH executables whose name starts with partial.

        Does not query Nushell's own command scope table: doing so
        would require a live process (there isn't one) and syntax that
        could not be verified without a local nu install. PATH-based
        completion needs neither and mirrors what bash's `compgen -c`
        effectively returns for external commands.
        """
        prefix = partial
        matches: set[str] = set()
        for directory in os.environ.get("PATH", "").split(os.pathsep):
            try:
                entries = await asyncio.to_thread(os.listdir, directory)
            except OSError:
                continue
            for name in entries:
                if not name.startswith(prefix):
                    continue
                full = os.path.join(directory, name)
                if os.access(full, os.X_OK) and not os.path.isdir(full):
                    matches.add(name)
        return sorted(matches)[:20]

    async def can_parse(self, raw: str) -> bool:
        """Return True if `nu-check` accepts raw as valid Nushell syntax.

        raw is written to a temp file and checked via `nu-check
        <path>`, Nushell's documented parse-only validation command,
        rather than piping raw through $in: nu-check's own docs only
        confirm the file form unambiguously, and getting stdin-to-$in
        binding wrong in a -c script could not be verified without a
        local nu install.
        """

        def _check() -> bool:
            try:
                exe = self._resolve_exe()
            except RuntimeError:
                return False
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".nu", delete=False
                ) as f:
                    f.write(raw)
                    tmp_path = f.name
                try:
                    script = (
                        f"if (nu-check '{tmp_path}') "
                        "{ exit 0 } else { exit 1 }"
                    )
                    result = subprocess.run(
                        [exe, "--no-config-file", "-c", script],
                        capture_output=True,
                        timeout=_CHECK_TIMEOUT,
                    )
                finally:
                    Path(tmp_path).unlink(missing_ok=True)
                return result.returncode == 0
            except subprocess.TimeoutExpired:
                return False

        return await asyncio.to_thread(_check)

    async def render_prompt(self) -> str:
        """Return a synthesized `cwd> ` prompt.

        Does not evaluate the user's real $env.PROMPT_COMMAND: doing so
        correctly requires closure-introspection syntax
        (``describe``/``do`` on a possibly-unset config value) that
        could not be verified without a local nu install, and a wrong
        guess here would silently produce a garbled prompt rather than
        a loud failure. A plain `cwd> ` string is Nushell's own default
        prompt shape when no custom prompt is configured.
        """
        return f"{self._cwd}> "

    async def append_history(self, command: str) -> None:
        """Append command to agentsh's own hardened history file.

        Unlike bash/PowerShell, no mirror into Nushell's own native
        history file is offered: that file's format (plaintext vs
        SQLite) and location are configurable and changed defaults
        across nu versions, and neither could be confirmed without a
        local nu install. Silently mirroring into what might actually
        be a SQLite-backed history would do nothing useful, so this
        backend keeps its own file as the sole sink rather than guess.
        """
        try:
            await asyncio.to_thread(
                append_secure_line, self._history_path, command
            )
        except OSError:
            pass

    async def reset(self) -> None:
        """Kill the in-flight one-shot subprocess, if any, and wait for it.

        There is no persistent process to restart -- the next execute()
        call always spawns a fresh one -- so this only matters for
        abandoning a command that timed out mid-flight.
        """
        async with self._lock:
            proc = self._current_proc
            if proc is not None and proc.returncode is None:
                proc.kill()
                await proc.wait()

    async def close(self) -> None:
        """Terminate the in-flight one-shot subprocess, if any."""
        await self.reset()
