"""Persistent Fish shell backend.

Fish is not POSIX-compatible, so this backend cannot reuse bash's
wrapping string verbatim:

- The last exit status is ``$status``, not ``$?``.
- Grouping a command for a scoped redirect uses ``begin ... end``, not
  ``{ ...; }``; the redirect is attached after the closing ``end``.
- Command substitution is ``(cmd)`` or, quoted, ``"$(cmd)"``; the
  quoted form is used here so a cwd is captured as a single argument
  even if it contained newlines.
- ``config.fish`` is sourced for every fish invocation, interactive or
  not (unlike bash, which only sources ``.bashrc`` for interactive
  shells), so the persistent session is started with ``--no-config``
  to keep the sentinel protocol free of unrelated startup output.
"""

import asyncio
import os
import subprocess
import time
from pathlib import Path

from agentsh.history_security import append_secure_line
from agentsh.limits import read_capped_text, read_last_lines
from agentsh.models import CommandResult
from agentsh.shell._registry import register
from agentsh.shell.plugin._base import ProcessBackedShell, new_marker
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


@register("fish")
class FishShell(ProcessBackedShell):
    """Wraps a persistent fish subprocess; tracks cwd after every command."""

    def __init__(self) -> None:
        """Initialise shared process state and agentsh's own history path."""
        super().__init__()
        self._history_path = _default_history_path()

    async def _start_process(self) -> asyncio.subprocess.Process:
        """Start the fish subprocess.

        ``--no-config`` skips ``config.fish``, which fish would
        otherwise source even for this non-interactive, piped-stdin
        session; without it, arbitrary user startup output could land
        on stdout and corrupt the sentinel protocol.
        """
        return await asyncio.create_subprocess_exec(
            "fish",
            "--no-config",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )

    async def execute(self, command: str) -> CommandResult:
        """Execute a shell command.

        The command runs inside a ``begin ... end`` block so its
        stderr can be redirected as a whole, regardless of what the
        command itself does with its own redirects; ``$status``
        (fish's ``$?`` equivalent) is captured into a variable
        immediately after the block, before any other command can
        change it.

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
                "begin\n"
                f"{command}\n"
                f"end 2>{stderr_path}\n"
                "set __ec $status\n"
                f'printf "%s:%d:%s\\n" "{marker}" "$__ec" "$(pwd)"\n'
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
                result = subprocess.run(
                    ["fish", "--no-execute", "-c", raw],
                    capture_output=True,
                    timeout=1.0,
                )
                return result.returncode == 0
            except subprocess.TimeoutExpired:
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
            proc = await asyncio.create_subprocess_exec(
                "fish",
                "-c",
                f"cd {_fish_quote(self._cwd)}; fish_prompt",
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
