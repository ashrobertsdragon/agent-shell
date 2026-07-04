"""Persistent Bash shell backend."""

from __future__ import annotations

import asyncio
import os
import shlex
import subprocess
import tempfile
import time
from pathlib import Path

from agentsh.models import CommandResult

_SENTINEL = "__AGENTSH_DONE_8675309__"


class BashShell:
    """Wraps a persistent bash subprocess; tracks cwd after every command."""

    _process: asyncio.subprocess.Process | None
    _cwd: str
    _lock: asyncio.Lock

    def __init__(self) -> None:
        """Initialise with no subprocess; it is started lazily on first execute."""
        self._process = None
        self._cwd = os.getcwd()
        self._lock = asyncio.Lock()

    async def _ensure_started(self) -> asyncio.subprocess.Process:
        """Start the bash subprocess if it is not running."""
        if self._process is None or self._process.returncode is not None:
            self._process = await asyncio.create_subprocess_exec(
                "bash",
                "--noprofile",
                "--norc",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
        return self._process

    async def execute(self, command: str) -> CommandResult:
        """Execute a shell command and return stdout, stderr, exit code, and cwd."""
        async with self._lock:
            proc = await self._ensure_started()
            assert proc.stdin and proc.stdout

            fd, stderr_path = tempfile.mkstemp(prefix="agentsh_stderr_")
            os.close(fd)

            start = time.monotonic()
            wrapped = (
                f"exec 3>&2 2>{stderr_path}\n"
                f"{command}\n"
                f"__ec__=$?\n"
                f"exec 2>&3 3>&-\n"
                f'printf "%s:%d:%s\\n" "{_SENTINEL}" "$__ec__" "$(pwd)"\n'
            )
            proc.stdin.write(wrapped.encode())
            await proc.stdin.drain()

            chunks: list[str] = []
            exit_code = 1
            async for line in proc.stdout:
                decoded = line.decode(errors="replace")
                if decoded.startswith(f"{_SENTINEL}:"):
                    _, code_str, cwd = decoded.strip().split(":", 2)
                    exit_code = int(code_str)
                    self._cwd = cwd
                    break
                chunks.append(decoded)

            stderr_content = Path(stderr_path).read_text(errors="replace")
            Path(stderr_path).unlink(missing_ok=True)
            duration_ms = (time.monotonic() - start) * 1000

            return CommandResult(
                stdout="".join(chunks),
                stderr=stderr_content,
                exit_code=exit_code,
                duration_ms=duration_ms,
                cwd=self._cwd,
            )

    async def cwd(self) -> str:
        """Return the last tracked working directory."""
        return self._cwd

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
        """Return lines from $HISTFILE (default ~/.bash_history)."""
        histfile = os.environ.get("HISTFILE", str(Path.home() / ".bash_history"))
        try:
            lines = Path(histfile).read_text().splitlines()
            return lines[-limit:]
        except FileNotFoundError:
            return []

    async def complete(self, partial: str) -> list[str]:
        """Return up to 20 command completions via compgen."""
        result = await self.execute(
            f"compgen -c {shlex.quote(partial)} 2>/dev/null | head -20"
        )
        return result.stdout.splitlines()

    def can_parse(self, raw: str) -> bool:
        """Return True if bash -n accepts the input as valid syntax."""
        try:
            result = subprocess.run(
                ["bash", "-n", "-c", raw],
                capture_output=True,
                timeout=1.0,
            )
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            return False

    async def render_prompt(self) -> str:
        """Evaluate PS1 via bash -i so the user's prompt configuration is active.

        bash -i sources .bashrc. ${PS1@P} expands all prompt sequences including
        command substitutions (starship, powerline, etc.). \\001 and \\002 are
        readline non-printing markers that must be stripped before display.
        """
        try:
            result = subprocess.run(
                [
                    "bash",
                    "-i",
                    "-c",
                    f"cd {shlex.quote(self._cwd)} && printf '%s' \"${{PS1@P}}\"",
                ],
                capture_output=True,
                text=True,
                timeout=2.0,
                env=os.environ,
            )
            prompt = result.stdout.replace("\001", "").replace("\002", "")
            if prompt:
                return prompt
        except subprocess.TimeoutExpired:
            pass
        return f"{self._cwd}$ "

    async def append_history(self, command: str) -> None:
        """Append command to $HISTFILE (default ~/.bash_history)."""
        histfile = os.environ.get("HISTFILE", str(Path.home() / ".bash_history"))
        try:
            with open(histfile, "a") as f:
                f.write(command + "\n")
        except OSError:
            pass

    async def close(self) -> None:
        """Terminate the underlying bash subprocess."""
        if self._process and self._process.returncode is None:
            self._process.terminate()
            await self._process.wait()
