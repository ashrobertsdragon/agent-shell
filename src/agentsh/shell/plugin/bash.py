"""Persistent Bash shell backend."""

import asyncio
import os
import shlex
import subprocess
import time
import warnings
from pathlib import Path

from agentsh.history_security import append_secure_line, env_flag_enabled
from agentsh.limits import read_capped_text, read_last_lines
from agentsh.models import CommandResult
from agentsh.shell._registry import register
from agentsh.shell.plugin._base import ProcessBackedShell, new_marker
from agentsh.shell.plugin._stderr_file import (
    create_stderr_tempfile,
    discard_stderr_tempfile,
)
from agentsh.shell.plugin._stream import read_until_sentinel

_SENTINEL = "__AGENTSH_DONE_8675309__"

_PROMPT_TIMEOUT = 2.0

_HISTFILE_MIRROR_ENV = "AGENTSH_MIRROR_HISTFILE"


def _default_history_path() -> Path:
    """Return agentsh's own bash history file, beside its config."""
    return Path.home() / ".config" / "agentsh" / "bash_history"


def _append_line(path: str, line: str) -> None:
    """Append line plus a trailing newline to path, off the event loop."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


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


@register("bash")
class BashShell(ProcessBackedShell):
    """Wraps a persistent bash subprocess; tracks cwd after every command."""

    def __init__(self) -> None:
        """Initialise shared process state and agentsh's own history path."""
        super().__init__()
        self._history_path = _default_history_path()
        self._histfile_mirror_warned = False

    async def _start_process(self) -> asyncio.subprocess.Process:
        """Start the bash subprocess."""
        return await asyncio.create_subprocess_exec(
            "bash",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )

    async def execute(self, command: str) -> CommandResult:
        """Execute a shell command.

        Output is capped at MAX_OUTPUT_BYTES; a command producing more
        (or a single line exceeding asyncio's internal buffer limit) is
        truncated with a marker rather than buffered unboundedly or
        crashing the process. See read_until_sentinel.

        Args:
            command (str): The shell command to run.

        Returns:
            CommandResult: The command stdout, stderr, exit code, and cwd.
        """
        async with self._lock:
            proc = await self.process

            stderr_path = await asyncio.to_thread(create_stderr_tempfile)

            start = time.monotonic()
            marker = new_marker(_SENTINEL)
            wrapped = (
                f"exec 3>&2 2>{stderr_path}\n"
                f"{command}\n"
                f"__ec__=$?\n"
                f"exec 2>&3 3>&-\n"
                f'printf "%s:%d:%s\\n" "{marker}" "$__ec__" "$(pwd)"\n'
            )
            exit_code: int
            try:
                if not proc.stdin:
                    raise ChildProcessError
                proc.stdin.write(wrapped.encode())
                await proc.stdin.drain()
                if not proc.stdout:
                    raise ChildProcessError
                stdout_content, sentinel_line = await read_until_sentinel(
                    proc.stdout, f"{marker}:"
                )
                parsed = (
                    _parse_sentinel(sentinel_line, marker)
                    if sentinel_line
                    else None
                )
                if parsed is None:
                    raise ChildProcessError
                exit_code, self._cwd = parsed

                stderr_content = await asyncio.to_thread(
                    read_capped_text, stderr_path
                )
                duration_ms = (time.monotonic() - start) * 1000

                return CommandResult(
                    stdout=stdout_content,
                    stderr=stderr_content,
                    exit_code=exit_code,
                    duration_ms=duration_ms,
                    cwd=self._cwd,
                )
            except ChildProcessError:
                stderr_content = await asyncio.to_thread(
                    read_capped_text, stderr_path
                )
                duration_ms = (time.monotonic() - start) * 1000
                return CommandResult(
                    stdout="",
                    stderr=stderr_content,
                    exit_code=proc.returncode or 1,
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
        """Return the last `limit` lines of agentsh's own bash history file."""
        try:
            return await asyncio.to_thread(
                read_last_lines, self._history_path, limit
            )
        except FileNotFoundError:
            return []

    async def complete(self, partial: str) -> list[str]:
        """Return up to 20 command completions via compgen."""
        result = await self.execute(
            f"compgen -c {shlex.quote(partial)} 2>/dev/null | head -20"
        )
        return result.stdout.splitlines()

    async def can_parse(self, raw: str) -> bool:
        """Return True if bash -n accepts the input as valid syntax."""

        def _check() -> bool:
            try:
                result = subprocess.run(
                    ["bash", "-n", "-c", raw],
                    capture_output=True,
                    timeout=1.0,
                )
                return result.returncode == 0
            except subprocess.TimeoutExpired:
                return False

        return await asyncio.to_thread(_check)

    async def render_prompt(self) -> str:
        """Evaluate PS1 via bash -i so the prompt configuration is active.

        bash -i sources .bashrc. ${PS1@P} expands all prompt sequences including
        command substitutions (starship, powerline, etc.).
        """
        fallback = f"{self._cwd}$ "
        try:
            proc = await asyncio.create_subprocess_exec(
                "bash",
                "-i",
                "-c",
                f"cd {shlex.quote(self._cwd)} && printf '%s' \"${{PS1@P}}\"",
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
        prompt = (
            stdout.decode(errors="replace")
            .replace("\001", "")
            .replace("\002", "")
        )
        return prompt or fallback

    async def append_history(self, command: str) -> None:
        """Append command to agentsh's own hardened history file.

        This is the only sink written by default; bash's own
        ``$HISTFILE`` is left untouched so agentsh never introduces a
        second, unhardened copy of plaintext command history. Set
        ``AGENTSH_MIRROR_HISTFILE=1`` to additionally mirror entries
        into ``$HISTFILE`` for native bash history integration; doing
        so duplicates commands (which may contain inline secrets) into
        a file agentsh does not harden, so a one-time warning is
        emitted per shell instance when the mirror is active.
        """
        try:
            await asyncio.to_thread(
                append_secure_line, self._history_path, command
            )
        except OSError:
            pass

        if not env_flag_enabled(_HISTFILE_MIRROR_ENV):
            return

        if not self._histfile_mirror_warned:
            warnings.warn(
                "AGENTSH_MIRROR_HISTFILE is set: commands are being "
                "duplicated into $HISTFILE, which agentsh does not "
                "harden to 0o600; secrets typed at the prompt may be "
                "persisted there world-readable.",
                stacklevel=2,
            )
            self._histfile_mirror_warned = True

        histfile = os.environ.get(
            "HISTFILE", str(Path.home() / ".bash_history")
        )
        try:
            await asyncio.to_thread(_append_line, histfile, command)
        except OSError:
            pass
