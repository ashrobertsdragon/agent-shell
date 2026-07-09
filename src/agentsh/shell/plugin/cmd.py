"""Persistent Windows CMD shell backend."""

import asyncio
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from agentsh.history_security import append_secure_line
from agentsh.limits import read_capped_text, read_last_lines
from agentsh.models import CommandResult
from agentsh.shell._registry import register
from agentsh.shell.plugin._base import ProcessBackedShell, new_marker
from agentsh.shell.plugin._stream import read_until_sentinel

_SENTINEL = "__AGENTSH_CMD_DONE_8675309__"

_BUILTINS = (
    "assoc",
    "call",
    "cd",
    "cls",
    "color",
    "copy",
    "date",
    "del",
    "dir",
    "echo",
    "endlocal",
    "erase",
    "exit",
    "for",
    "ftype",
    "goto",
    "if",
    "md",
    "mkdir",
    "mklink",
    "move",
    "path",
    "pause",
    "popd",
    "prompt",
    "pushd",
    "rd",
    "rem",
    "ren",
    "rename",
    "rmdir",
    "set",
    "setlocal",
    "shift",
    "start",
    "time",
    "title",
    "type",
    "ver",
    "verify",
    "vol",
)

_PROMPT_CODES = {
    "G": ">",
    "L": "<",
    "B": "|",
    "Q": "=",
    "A": "&",
    "S": " ",
    "_": "\n",
    "$": "$",
}


def _expand_prompt(template: str, cwd: str) -> str:
    """Expand common cmd PROMPT $-codes; unknown codes are dropped."""

    def _sub(match: re.Match[str]) -> str:
        code = match.group(1).upper()
        return cwd if code == "P" else _PROMPT_CODES.get(code, "")

    return re.sub(r"\$(.)", _sub, template)


def _parse_sentinel(line: str, marker: str) -> tuple[int, str] | None:
    r"""Return (exit_code, cwd) if line is an exact sentinel match for marker.

    maxsplit=2 keeps cwd intact as the final field, so drive-letter
    colons in ``%cd%`` (``C:\...``) are safe. The first field must equal
    marker exactly (not merely start with it), so command output that
    contains sentinel-like text cannot be mistaken for the real one.
    """
    parts = line.strip().split(":", 2)
    if len(parts) != 3 or parts[0] != marker:
        return None
    try:
        return int(parts[1]), parts[2]
    except ValueError:
        return None


def _complete_from_path(partial: str, path: str, pathext: str) -> list[str]:
    """Return PATH executables whose stem matches partial case-insensitively.

    Args:
        partial (str): The command prefix to match.
        path (str): A PATH-style list of directories.
        pathext (str): A PATHEXT-style list of executable extensions.

    Returns:
        list[str]: Sorted, deduplicated stems, capped at 20.
    """
    exts = {ext.lower() for ext in pathext.split(";") if ext}
    prefix = partial.lower()
    matches = {
        entry.stem
        for directory in path.split(os.pathsep)
        for entry in Path(directory).glob("*")
        if entry.stem.lower().startswith(prefix)
        and entry.suffix.lower() in exts
    }
    return sorted(matches)[:20]


def _default_history_path() -> Path:
    """Return agentsh's own cmd history file, beside its config."""
    return Path.home() / ".config" / "agentsh" / "cmd_history"


def _detect_clink() -> str | None:
    """Return the clink executable path if clink is on PATH."""
    return shutil.which("clink")


@register("cmd")
class CmdShell(ProcessBackedShell):
    """Wraps a persistent cmd.exe subprocess; tracks cwd per command.

    cmd.exe has no persistent history, so agentsh keeps its own history
    file as the source of truth; when clink is installed, appends are
    additionally mirrored into clink via ``clink history add`` so they
    appear in the user's interactive sessions.

    The subprocess runs with /Q (echo off) to suppress command echo on
    piped stdin; residual prompt lines on some systems are a known risk
    pending real-Windows verification.
    """

    def __init__(self) -> None:
        """Initialise subprocess state, history path, and clink lookup."""
        super().__init__()
        self._exe: str | None = None
        self._history_path = _default_history_path()
        self._clink = _detect_clink()

    async def _start_process(self) -> asyncio.subprocess.Process:
        """Start the cmd subprocess and switch it to UTF-8."""
        if self._exe is None:
            self._exe = (
                os.environ.get("COMSPEC") or shutil.which("cmd") or "cmd"
            )
        proc = await asyncio.create_subprocess_exec(
            self._exe,
            "/Q",
            "/K",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        if proc.stdin:
            proc.stdin.write(b"chcp 65001>nul\r\n")
            await proc.stdin.drain()
        return proc

    async def execute(self, command: str) -> CommandResult:
        """Execute a cmd command.

        The command is wrapped in parentheses so stderr redirection
        covers compound commands; the sentinel is sent as a separate
        stdin line so %errorlevel% and %cd% expand after the command.

        Output is capped at MAX_OUTPUT_BYTES; a command producing more
        (or a single line exceeding asyncio's internal buffer limit) is
        truncated with a marker rather than buffered unboundedly or
        crashing the process. See read_until_sentinel.

        Args:
            command (str): The cmd command to run.

        Returns:
            CommandResult: The command stdout, stderr, exit code, and cwd.
        """
        async with self._lock:
            proc = await self.process

            fd, stderr_path = tempfile.mkstemp(prefix="agentsh_stderr_")
            os.close(fd)

            start = time.monotonic()
            marker = new_marker(_SENTINEL)
            wrapped = (
                f'({command} ) 2>"{stderr_path}"\r\n'
                f"echo {marker}:%errorlevel%:%cd%\r\n"
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
                    proc.stdout,
                    f"{marker}:",
                    transform=lambda decoded: decoded.replace("\r\n", "\n"),
                )
                parsed = (
                    _parse_sentinel(sentinel_line, marker)
                    if sentinel_line
                    else None
                )
                if parsed is None:
                    raise ChildProcessError
                exit_code, self._cwd = parsed

                stderr_content = read_capped_text(stderr_path)
                duration_ms = (time.monotonic() - start) * 1000

                return CommandResult(
                    stdout=stdout_content,
                    stderr=stderr_content,
                    exit_code=exit_code,
                    duration_ms=duration_ms,
                    cwd=self._cwd,
                )
            except ChildProcessError:
                stderr_content = read_capped_text(stderr_path)
                duration_ms = (time.monotonic() - start) * 1000
                return CommandResult(
                    stdout="",
                    stderr=stderr_content,
                    exit_code=proc.returncode or 1,
                    duration_ms=duration_ms,
                    cwd=self._cwd,
                )
            finally:
                Path(stderr_path).unlink(missing_ok=True)

    async def env(self) -> dict[str, str]:
        """Return the subprocess environment by running set.

        Lines starting with ``=`` (cmd's hidden per-drive variables)
        are skipped.
        """
        result = await self.execute("set")
        env: dict[str, str] = {}
        for line in result.stdout.splitlines():
            if "=" in line and not line.startswith("="):
                k, _, v = line.partition("=")
                env[k] = v
        return env

    async def history(self, limit: int = 100) -> list[str]:
        """Return the last `limit` lines of agentsh's own cmd history file."""
        try:
            return read_last_lines(self._history_path, limit)
        except FileNotFoundError:
            return []

    async def complete(self, partial: str) -> list[str]:
        """Return up to 20 matches from cmd builtins and PATH executables."""
        path_matches = _complete_from_path(
            partial,
            os.environ.get("PATH", ""),
            os.environ.get("PATHEXT", ".COM;.EXE;.BAT;.CMD"),
        )
        prefix = partial.lower()
        builtin_matches = [b for b in _BUILTINS if b.startswith(prefix)]
        return sorted(set(builtin_matches) | set(path_matches))[:20]

    async def can_parse(self, raw: str) -> bool:
        """Return True unconditionally; cmd has no syntax-check mode."""
        return True

    async def render_prompt(self) -> str:
        """Expand the PROMPT env var (default $P$G) against tracked cwd."""
        prompt = _expand_prompt(os.environ.get("PROMPT", "$P$G"), self._cwd)
        return prompt or f"{self._cwd}>"

    async def append_history(self, command: str) -> None:
        """Append command to the own hardened history file, mirroring to clink.

        The own history file is created (or re-hardened, if it already
        exists) with mode 0o600 since it is agentsh's own file and no
        other program depends on its permissions. The clink mirror
        routes through ``clink history add`` so entries land in
        clink's master history safely; any clink failure is swallowed.
        """
        try:
            append_secure_line(self._history_path, command)
        except OSError:
            pass

        clink = self._clink
        if clink is None:
            return

        def _mirror() -> None:
            try:
                subprocess.run(
                    [clink, "history", "add", command],
                    capture_output=True,
                    timeout=2.0,
                )
            except (OSError, subprocess.SubprocessError):
                pass

        await asyncio.to_thread(_mirror)
