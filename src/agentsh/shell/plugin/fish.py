"""Fish shell backend backed by one-shot processes.

Fish refuses to run any statement fed over a pipe until stdin reaches
EOF: writing a command to a live fish subprocess's stdin does not
execute it, even with ``-i`` forced, until the pipe is closed
(confirmed against a real fish binary). That rules out the persistent,
sentinel-over-a-long-lived-pipe protocol every other backend in this
package uses (see ``_base.ProcessBackedShell``): there is no way to
keep one ``fish`` process alive across calls and feed it commands one
at a time the way bash/cmd/PowerShell are fed.

Instead, FishShell spawns a fresh, non-interactive ``fish --no-config
-c <script>`` process for every ``execute()`` call, mirroring
NuShellShell's design (see ``nushell.py``). State that must survive
between calls -- only the working directory, here -- is threaded
through explicitly: each call starts fish with ``cwd=self._cwd`` and
reads the resulting directory back out of a sentinel line.

Fish is also not POSIX-compatible, so the wrapper script itself still
differs from bash's:

- The last exit status is ``$status``, not ``$?``.
- Command substitution is ``(cmd)`` or, quoted, ``"$(cmd)"``; the
  quoted form is used here so a cwd is captured as a single argument
  even if it contained newlines.

Because each call now gets its own dedicated process with stderr fixed
to a file for the whole process lifetime (like nu's ``_spawn``), the
old persistent-backend's ``begin ... end 2>path`` scoped-redirect
trick is no longer needed: the command runs directly and its stderr
naturally lands in the per-call stderr file.
"""

import asyncio
import os
import shutil
import subprocess
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

_SENTINEL = "__AGENTSH_FISH_DONE_8675309__"

_PROMPT_TIMEOUT = 2.0


def _default_history_path() -> Path:
    """Return agentsh's own fish history file, beside its config."""
    return Path.home() / ".config" / "agentsh" / "fish_history"


def _parse_sentinel(line: str, marker: str) -> tuple[int, str] | None:
    """Return (exit_code, cwd) if line is an exact sentinel match for marker.

    The comparison is against the whole line (split into exactly three
    fields), not a prefix or substring, so command output that merely
    contains sentinel-like text cannot be mistaken for the real one.
    """
    parts = line.rstrip("\r\n").split(":", 2)
    if len(parts) != 3 or parts[0] != marker:
        return None
    try:
        return int(parts[1]), parts[2]
    except ValueError:
        return None


def _fish_quote(value: str) -> str:
    r"""Return value as a fish single-quoted string literal.

    Fish's single-quoted strings only treat ``\'`` and ``\\`` as
    escapes (everything else is literal), unlike POSIX sh where
    nothing is special inside single quotes.
    """
    escaped = value.replace("\\", "\\\\").replace("'", "\\'")
    return f"'{escaped}'"


def _wrap_command(command: str, marker: str) -> str:
    """Wrap command so the exit status and final cwd are always reported.

    ``$status`` is captured into a variable immediately after the
    command, before any other statement (including the printf itself)
    can change it. Fish does not abort a ``-c`` script on a nonzero
    exit unless ``status --is-command-substitution``-style opt-in
    behaviour is configured, so -- like the original persistent-backend
    wrapper -- no try/catch is needed to guarantee the sentinel line is
    always printed.
    """
    return (
        f"{command}\n"
        "set __ec $status\n"
        f'printf "%s:%d:%s\\n" "{marker}" "$__ec" "$(pwd)"\n'
    )


@register("fish")
class FishShell:
    """Wraps fish as a sequence of one-shot ``fish -c`` subprocesses.

    Does not subclass ProcessBackedShell: that base class's whole
    purpose is managing one long-lived subprocess's lifecycle (lazy
    start, restart on exit/desync, forced reset), and fish has no
    long-lived subprocess to manage here -- see the module docstring
    for why a persistent, pipe-fed fish process isn't viable. `reset`
    and `close` still do something real (killing an in-flight one-shot
    process on cancellation) but there is no "restart on next use"
    concept to inherit.
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
        """Return the fish executable path, resolving and caching it lazily.

        Raises:
            RuntimeError: If no fish executable is found on PATH.
        """
        if self._exe is None:
            exe = shutil.which("fish")
            if exe is None:
                raise RuntimeError("no fish executable found")
            self._exe = exe
        return self._exe

    async def _spawn(
        self, exe: str, script: str, stderr_path: str
    ) -> asyncio.subprocess.Process:
        """Start a one-shot fish subprocess with stderr redirected to a file.

        Unlike the persistent backends (which toggle stderr redirection
        mid-script because many commands share one long-lived process),
        this process runs exactly one command, so its stderr fd can be
        fixed for the whole process at spawn time.
        """
        stderr_file = await asyncio.to_thread(open, stderr_path, "wb")
        try:
            return await asyncio.create_subprocess_exec(
                exe,
                "--no-config",
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
        """Execute a fish command in a fresh, non-interactive fish process.

        Output is capped at MAX_OUTPUT_BYTES via the shared
        read_until_sentinel helper. If the sentinel line never arrives
        (e.g. the command called ``exit`` directly), the process's own
        exit code is used and cwd is left unchanged, but any output
        collected before that point is still returned.

        Args:
            command (str): The shell command to run.

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
        """Return the subprocess environment by running env."""
        result = await self.execute("env")
        env: dict[str, str] = {}
        for line in result.stdout.splitlines():
            if "=" in line:
                k, _, v = line.partition("=")
                env[k] = v
        return env

    async def history(self, limit: int = 100) -> list[str]:
        """Return the last `limit` lines of agentsh's own fish history file."""
        try:
            return await asyncio.to_thread(
                read_last_lines, self._history_path, limit
            )
        except FileNotFoundError:
            return []

    async def complete(self, partial: str) -> list[str]:
        """Return up to 20 command completions via fish's own completer.

        ``complete -C`` runs fish's completion engine against a
        candidate commandline and prints matches one per line, each
        optionally followed by a tab and a description; only the
        completion text itself is kept.
        """
        result = await self.execute(f"complete -C{_fish_quote(partial)}")
        names = [
            line.split("\t", 1)[0]
            for line in result.stdout.splitlines()
            if line
        ]
        return names[:20]

    async def can_parse(self, raw: str) -> bool:
        """Return True if fish --no-execute accepts raw as valid syntax."""

        def _check() -> bool:
            try:
                exe = self._resolve_exe()
            except RuntimeError:
                return False
            try:
                result = subprocess.run(
                    [exe, "--no-execute", "-c", raw],
                    capture_output=True,
                    timeout=1.0,
                )
                return result.returncode == 0
            except (subprocess.TimeoutExpired, OSError):
                return False

        return await asyncio.to_thread(_check)

    async def render_prompt(self) -> str:
        """Evaluate fish_prompt with config.fish loaded.

        Unlike bash, fish sources config.fish for every invocation
        regardless of interactivity, so no ``-i`` flag is needed to
        pick up prompt customisations (starship, Tide, etc.) defined
        there; fish_prompt is fish's direct equivalent of evaluating
        bash's PS1.
        """
        fallback = f"{self._cwd}> "
        try:
            exe = self._resolve_exe()
        except RuntimeError:
            return fallback
        try:
            proc = await asyncio.create_subprocess_exec(
                exe,
                "-c",
                "fish_prompt",
                cwd=self._cwd,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
                env=os.environ,
            )
        except OSError:
            return fallback
        try:
            stdout, _ = await asyncio.wait_for(
                proc.communicate(), timeout=_PROMPT_TIMEOUT
            )
        except TimeoutError:
            proc.kill()
            await proc.wait()
            return fallback
        prompt = stdout.decode(errors="replace").strip("\r\n")
        return prompt or fallback

    async def append_history(self, command: str) -> None:
        """Append command to agentsh's own hardened history file.

        This is the only sink written: fish's native history
        (``~/.local/share/fish/fish_history``) uses a structured,
        timestamped block format rather than one command per line, so
        appending a plain line to it (as the bash/PowerShell backends'
        opt-in mirror does for their plain-text native histories)
        would not merely duplicate an entry but corrupt the file. No
        native-history mirror is offered for fish.
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
