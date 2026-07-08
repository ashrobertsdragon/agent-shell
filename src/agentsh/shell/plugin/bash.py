"""Persistent Bash shell backend."""

import asyncio
import os
import shlex
import subprocess
import tempfile
import time
from pathlib import Path

from agentsh.models import CommandResult
from agentsh.shell._registry import register
from agentsh.shell.plugin._base import ProcessBackedShell, new_marker

_SENTINEL = "__AGENTSH_DONE_8675309__"

_PROMPT_TIMEOUT = 2.0


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

        Args:
            command (str): The shell command to run.

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
                f"exec 3>&2 2>{stderr_path}\n"
                f"{command}\n"
                f"__ec__=$?\n"
                f"exec 2>&3 3>&-\n"
                f'printf "%s:%d:%s\\n" "{marker}" "$__ec__" "$(pwd)"\n'
            )
            chunks: list[str] = []
            exit_code: int
            try:
                if not proc.stdin:
                    raise ChildProcessError
                proc.stdin.write(wrapped.encode())
                await proc.stdin.drain()
                if not proc.stdout:
                    raise ChildProcessError
                async for line in proc.stdout:
                    decoded = line.decode(errors="replace")
                    parsed = _parse_sentinel(decoded, marker)
                    if parsed is not None:
                        exit_code, self._cwd = parsed
                        break
                    chunks.append(decoded)
                else:
                    raise ChildProcessError

                stderr_content = Path(stderr_path).read_text(errors="replace")
                duration_ms = (time.monotonic() - start) * 1000

                return CommandResult(
                    stdout="".join(chunks),
                    stderr=stderr_content,
                    exit_code=exit_code,
                    duration_ms=duration_ms,
                    cwd=self._cwd,
                )
            except ChildProcessError:
                stderr_content = Path(stderr_path).read_text(errors="replace")
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
        histfile = os.environ.get(
            "HISTFILE", str(Path.home() / ".bash_history")
        )
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
        """Append command to $HISTFILE (default ~/.bash_history)."""
        histfile = os.environ.get(
            "HISTFILE", str(Path.home() / ".bash_history")
        )
        try:
            with open(histfile, "a") as f:
                f.write(command + "\n")
        except OSError:
            pass
