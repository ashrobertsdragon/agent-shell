"""Shared process lifecycle and sentinel helpers for shell backends.

All persistent shell backends (bash, cmd, powershell) drive a subprocess
over stdin/stdout and detect command completion by writing a sentinel
line after the user's command and reading stdout until that line
appears. Two properties of that protocol matter for correctness:

- The sentinel must include a per-call nonce so command output cannot
  forge completion by printing a sentinel-lookalike line.
- On a timeout that abandons an in-flight command, the subprocess must
  be killed and restarted rather than left running, or a future call
  reads output left over from the abandoned command.
"""

import asyncio
import os
import uuid
from abc import ABC, abstractmethod


def new_marker(sentinel: str) -> str:
    """Return a fresh, per-call sentinel marker combining sentinel and nonce.

    The nonce is joined with `_`, not `:`, so the marker itself never
    contains a colon; this keeps the ``marker:exit_code:cwd`` sentinel
    line splittable with a fixed maxsplit while comparing the marker
    field for exact equality.
    """
    return f"{sentinel}_{uuid.uuid4().hex}"


class ProcessBackedShell(ABC):
    """Owns the lifecycle of a persistent, sentinel-protocol subprocess.

    Subclasses implement `_start_process` plus their own command wrapping
    and sentinel parsing; this base handles lazy start, restart on exit
    or detected desync, and forced restart via `reset`.
    """

    def __init__(self) -> None:
        """Initialise shared process, lock, cwd, and desync state."""
        self._process: asyncio.subprocess.Process | None = None
        self._cwd = os.getcwd()
        self._lock = asyncio.Lock()
        self._desynced = False

    @abstractmethod
    async def _start_process(self) -> asyncio.subprocess.Process:
        """Start and return a fresh subprocess for this shell."""

    @property
    async def process(self) -> asyncio.subprocess.Process:
        """Return the live subprocess, restarting it if dead or desynced."""
        if (
            self._process is None
            or self._process.returncode is not None
            or self._desynced
        ):
            self._process = await self._start_process()
            self._desynced = False
        return self._process

    @property
    def cwd(self) -> str:
        """Return the last tracked working directory."""
        return self._cwd

    async def reset(self) -> None:
        """Kill the current subprocess and force a restart on next use.

        Called when a caller knows the shell may be desynced, e.g. after
        a context-collection timeout abandoned an in-flight command
        whose reader was cancelled mid-stream: the subprocess is still
        running and its eventual sentinel line would otherwise corrupt
        the next `execute` call.
        """
        async with self._lock:
            proc = self._process
            if proc is not None and proc.returncode is None:
                proc.kill()
                await proc.wait()
            self._desynced = True

    async def close(self) -> None:
        """Terminate the underlying subprocess."""
        if self._process and self._process.returncode is None:
            self._process.terminate()
            await self._process.wait()
