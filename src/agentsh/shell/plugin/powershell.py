"""Persistent PowerShell shell backend."""

import asyncio
import base64
import os
import shutil
import subprocess
import tempfile
import time
import warnings
from pathlib import Path

from agentsh.history_security import append_secure_line, env_flag_enabled
from agentsh.limits import read_capped_text
from agentsh.models import CommandResult
from agentsh.shell._registry import register
from agentsh.shell.plugin._base import ProcessBackedShell, new_marker
from agentsh.shell.plugin._stream import read_until_sentinel

_SENTINEL = "__AGENTSH_PS_DONE_8675309__"

_PROMPT_TIMEOUT = 3.0

_PSREADLINE_MIRROR_ENV = "AGENTSH_MIRROR_PSREADLINE_HISTORY"


def _default_history_path() -> Path:
    """Return agentsh's own PowerShell history file, beside its config."""
    return Path.home() / ".config" / "agentsh" / "powershell_history"


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


def _parse_sentinel(line: str, marker: str) -> tuple[int, str] | None:
    r"""Return (exit_code, cwd) if line is an exact sentinel match for marker.

    maxsplit=2 keeps cwd intact as the final field, so drive-letter
    colons in Windows paths (``C:\...``) are safe. The first field must
    equal marker exactly (not merely start with it), so command output
    that contains sentinel-like text cannot be mistaken for the real one.
    """
    parts = line.strip().split(":", 2)
    if len(parts) != 3 or parts[0] != marker:
        return None
    try:
        return int(parts[1]), parts[2]
    except ValueError:
        return None


def _ps_quote(value: str) -> str:
    """Return value as a PowerShell single-quoted string literal."""
    escaped = value.replace("'", "''")
    return f"'{escaped}'"


def _wrap_command(
    command: str, stderr_path: str, marker: str = _SENTINEL
) -> str:
    """Wrap a command to capture stderr and emit the sentinel line.

    The command is base64-encoded so any content is a single safe
    statement; the stderr path is quoted as a literal so paths with
    single quotes cannot break the wrapper syntax. marker is the
    per-call sentinel (base sentinel plus a random nonce) so command
    output cannot forge completion.
    """
    b64 = base64.b64encode(command.encode()).decode("ascii")
    stderr_literal = _ps_quote(stderr_path)
    return (
        "$global:LASTEXITCODE = 0\n"
        "$__cmd = [Text.Encoding]::UTF8.GetString("
        f"[Convert]::FromBase64String('{b64}'))\n"
        f"try {{ Invoke-Expression $__cmd 2>>{stderr_literal}; "
        "$__ok = $? } catch { $_ | Out-String | "
        f"Add-Content {stderr_literal}; $__ok = $false }}\n"
        "$__ec = if ($__ok -and -not $LASTEXITCODE) { 0 } "
        "elseif ($LASTEXITCODE) { $LASTEXITCODE } else { 1 }\n"
        f'"{marker}:$__ec:$((Get-Location).Path)"\n'
    )


@register("powershell")
class PowerShellShell(ProcessBackedShell):
    """Wraps a persistent PowerShell subprocess; tracks cwd per command.

    Commands are base64-encoded and run via Invoke-Expression so every
    line written to the subprocess stdin is a complete statement, which
    sidesteps the blank-line terminator quirk of ``-Command -`` for
    multi-line input.
    """

    def __init__(self) -> None:
        """Initialise subprocess state without resolving an executable."""
        super().__init__()
        self._exe: str | None = None
        self._history_path = _default_history_path()
        self._psreadline_mirror_warned = False

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
            marker = new_marker(_SENTINEL)
            wrapped = _wrap_command(command, stderr_path, marker)
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
        """Return lines from agentsh's own PowerShell history file."""
        try:
            lines = self._history_path.read_text().splitlines()
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
        fallback = f"PS {self._cwd}> "
        try:
            exe = _resolve_powershell()
            proc = await asyncio.create_subprocess_exec(
                exe,
                "-NoLogo",
                "-Command",
                f"Set-Location {_ps_quote(self._cwd)}; prompt",
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
        except (RuntimeError, OSError):
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

        This is the only sink written by default; PowerShell's own
        PSReadLine ``ConsoleHost_history.txt`` is left untouched so
        agentsh never introduces a second, unhardened copy of
        plaintext command history. Set
        ``AGENTSH_MIRROR_PSREADLINE_HISTORY=1`` to additionally mirror
        entries there for native PowerShell history integration; doing
        so duplicates commands (which may contain inline secrets) into
        a file agentsh does not harden, so a one-time warning is
        emitted per shell instance when the mirror is active.
        """
        try:
            append_secure_line(self._history_path, command)
        except OSError:
            pass

        if not env_flag_enabled(_PSREADLINE_MIRROR_ENV):
            return

        if not self._psreadline_mirror_warned:
            warnings.warn(
                "AGENTSH_MIRROR_PSREADLINE_HISTORY is set: commands "
                "are being duplicated into PSReadLine's "
                "ConsoleHost_history.txt, which agentsh does not "
                "harden to 0o600; secrets typed at the prompt may be "
                "persisted there world-readable.",
                stacklevel=2,
            )
            self._psreadline_mirror_warned = True

        path = _psreadline_history_path()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a") as f:
                f.write(command + "\n")
        except OSError:
            pass
