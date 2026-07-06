"""Persistent PowerShell shell backend."""

import asyncio
import base64
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from agentsh.models import CommandResult
from agentsh.shell._registry import register

_SENTINEL = "__AGENTSH_PS_DONE_8675309__"

_INIT = (
    "[Console]::OutputEncoding = [Text.Encoding]::UTF8; "
    "$OutputEncoding = [Text.Encoding]::UTF8; "
    "$ProgressPreference = 'SilentlyContinue'\n"
)


def _resolve_powershell() -> str:
    """Return the PowerShell executable path, preferring pwsh.

    Raises:
        RuntimeError: If neither pwsh nor powershell is on PATH.
    """
    exe = shutil.which("pwsh") or shutil.which("powershell")
    if exe is None:
        raise RuntimeError("no PowerShell executable found")
    return exe


def _psreadline_history_path(platform: str = os.name) -> Path:
    """Return the default PSReadLine history file path for a platform.

    Computed in Python rather than queried via ``Get-PSReadLineOption``
    because PSReadLine is not loaded in a -NoProfile -Command - session
    and history reads sit inside the context-building time budget.
    """
    if platform == "nt":
        appdata = os.environ.get(
            "APPDATA", str(Path.home() / "AppData" / "Roaming")
        )
        return (
            Path(appdata)
            / "Microsoft"
            / "Windows"
            / "PowerShell"
            / "PSReadLine"
            / "ConsoleHost_history.txt"
        )
    data = os.environ.get(
        "XDG_DATA_HOME", str(Path.home() / ".local" / "share")
    )
    return Path(data) / "powershell" / "PSReadLine" / "ConsoleHost_history.txt"


def _parse_sentinel(line: str) -> tuple[int, str]:
    r"""Parse a ``SENTINEL:code:cwd`` line into (exit_code, cwd).

    maxsplit=2 keeps cwd intact as the final field, so drive-letter
    colons in Windows paths (``C:\...``) are safe.
    """
    _, code, cwd = line.strip().split(":", 2)
    return int(code), cwd


def _ps_quote(value: str) -> str:
    """Return value as a PowerShell single-quoted string literal."""
    escaped = value.replace("'", "''")
    return f"'{escaped}'"


@register("powershell")
class PowerShellShell:
    """Wraps a persistent PowerShell subprocess; tracks cwd per command.

    Commands are base64-encoded and run via Invoke-Expression so every
    line written to the subprocess stdin is a complete statement, which
    sidesteps the blank-line terminator quirk of ``-Command -`` for
    multi-line input.
    """

    def __init__(self) -> None:
        """Initialise subprocess state without resolving an executable."""
        self._process: asyncio.subprocess.Process | None = None
        self._cwd = os.getcwd()
        self._lock = asyncio.Lock()
        self._exe: str | None = None

    async def _start_process(self) -> asyncio.subprocess.Process:
        """Start the PowerShell subprocess and apply session init."""
        if self._exe is None:
            self._exe = _resolve_powershell()
        proc = await asyncio.create_subprocess_exec(
            self._exe,
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            "-",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        if proc.stdin:
            proc.stdin.write(_INIT.encode())
            await proc.stdin.drain()
        return proc

    @property
    async def process(self) -> asyncio.subprocess.Process:
        """Property for subprocess instance."""
        if self._process is None or self._process.returncode is not None:
            self._process = await self._start_process()
        return self._process

    async def execute(self, command: str) -> CommandResult:
        """Execute a PowerShell command.

        Exit codes are best-effort: $LASTEXITCODE is pre-reset and used
        for native commands, $? covers cmdlet failure, and terminating
        errors caught by the catch block yield 1.

        Args:
            command (str): The PowerShell command to run.

        Returns:
            CommandResult: The command stdout, stderr, exit code, and cwd.
        """
        async with self._lock:
            proc = await self.process

            fd, stderr_path = tempfile.mkstemp(prefix="agentsh_stderr_")
            os.close(fd)

            start = time.monotonic()
            b64 = base64.b64encode(command.encode()).decode("ascii")
            wrapped = (
                "$global:LASTEXITCODE = 0\n"
                "$__cmd = [Text.Encoding]::UTF8.GetString("
                f"[Convert]::FromBase64String('{b64}'))\n"
                f"try {{ Invoke-Expression $__cmd 2>>'{stderr_path}'; "
                "$__ok = $? } catch { $_ | Out-String | "
                f"Add-Content '{stderr_path}'; $__ok = $false }}\n"
                "$__ec = if ($__ok -and -not $LASTEXITCODE) { 0 } "
                "elseif ($LASTEXITCODE) { $LASTEXITCODE } else { 1 }\n"
                f'"{_SENTINEL}:$__ec:$((Get-Location).Path)"\n'
            )
            chunks: list[str] = []
            exit_code = 1
            try:
                if not proc.stdin:
                    raise ChildProcessError
                proc.stdin.write(wrapped.encode())
                await proc.stdin.drain()
                if not proc.stdout:
                    raise ChildProcessError
                async for line in proc.stdout:
                    decoded = line.decode(errors="replace")
                    if decoded.startswith(f"{_SENTINEL}:"):
                        exit_code, self._cwd = _parse_sentinel(decoded)
                        break
                    chunks.append(decoded)

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

    @property
    def cwd(self) -> str:
        """Return the last tracked working directory."""
        return self._cwd

    async def env(self) -> dict[str, str]:
        """Return the subprocess environment via Get-ChildItem env:."""
        result = await self.execute(
            'Get-ChildItem env: | ForEach-Object { "$($_.Name)=$($_.Value)" }'
        )
        env: dict[str, str] = {}
        for line in result.stdout.splitlines():
            if "=" in line:
                k, _, v = line.partition("=")
                env[k] = v
        return env

    async def history(self, limit: int = 100) -> list[str]:
        """Return lines from the default PSReadLine history file."""
        try:
            lines = _psreadline_history_path().read_text().splitlines()
            return lines[-limit:]
        except FileNotFoundError:
            return []

    async def complete(self, partial: str) -> list[str]:
        """Return up to 20 completions via CommandCompletion."""
        script = (
            "[System.Management.Automation.CommandCompletion]::CompleteInput("
            f"{_ps_quote(partial)}, {len(partial)}, $null).CompletionMatches"
            " | Select-Object -First 20 -ExpandProperty CompletionText"
        )
        result = await self.execute(script)
        return result.stdout.splitlines()

    async def can_parse(self, raw: str) -> bool:
        """Return True if the PowerShell parser accepts raw as valid.

        raw is passed via stdin to avoid quoting issues; the timeout is
        generous because PowerShell cold-start is slow.
        """

        def _check() -> bool:
            try:
                exe = _resolve_powershell()
            except RuntimeError:
                return False
            script = (
                "$errs = $null; "
                "[System.Management.Automation.Language.Parser]::ParseInput("
                "[Console]::In.ReadToEnd(), [ref]$null, [ref]$errs)"
                " | Out-Null; "
                "exit [int]($errs.Count -gt 0)"
            )
            try:
                result = subprocess.run(
                    [
                        exe,
                        "-NoLogo",
                        "-NoProfile",
                        "-NonInteractive",
                        "-Command",
                        script,
                    ],
                    input=raw.encode(),
                    capture_output=True,
                    timeout=5.0,
                )
                return result.returncode == 0
            except (OSError, subprocess.SubprocessError):
                return False

        return await asyncio.to_thread(_check)

    async def render_prompt(self) -> str:
        """Evaluate the user's prompt function with the profile loaded.

        -NoProfile is deliberately omitted so custom prompt functions
        (oh-my-posh, starship, etc.) are active.
        """
        try:
            exe = _resolve_powershell()
            proc = await asyncio.create_subprocess_exec(
                exe,
                "-NoLogo",
                "-Command",
                f"Set-Location {_ps_quote(self._cwd)}; prompt",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=3.0)
            prompt = stdout.decode(errors="replace").strip("\r\n")
            if prompt:
                return prompt
        except (TimeoutError, RuntimeError, OSError):
            pass
        return f"PS {self._cwd}> "

    async def append_history(self, command: str) -> None:
        """Append command to the default PSReadLine history file."""
        path = _psreadline_history_path()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a") as f:
                f.write(command + "\n")
        except OSError:
            pass

    async def close(self) -> None:
        """Terminate the underlying PowerShell subprocess."""
        if self._process and self._process.returncode is None:
            self._process.terminate()
            await self._process.wait()
