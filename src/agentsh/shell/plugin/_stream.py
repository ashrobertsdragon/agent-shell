"""Bounded, crash-safe readline loop shared by the persistent shell backends.

asyncio.StreamReader.readline() wraps readuntil() and, on a
LimitOverrunError (a single line exceeding the stream's internal buffer
limit), converts it to a plain ValueError -- but only after clearing the
*entire* internal buffer when the newline isn't found exactly at the
overrun boundary, not just the oversized line. That risks silently
discarding subsequently-arrived data, including the sentinel line itself,
desyncing the shell's request/response protocol for the next command.

This helper bypasses readline() and calls readuntil() directly, handling
LimitOverrunError itself: it advances past exactly the reported overrun
bytes via readexactly() (which readuntil() guarantees are still sitting
in the buffer) and keeps reading, so no data is ever silently dropped.

It also caps total buffered output at MAX_OUTPUT_BYTES so a command that
emits gigabytes of ordinary output cannot exhaust memory or blow up an
LLM prompt, and applies truncate_text() as a final safety net since
decoding with errors="replace" can inflate byte length past a raw cap.
"""

import asyncio
from collections.abc import Callable

from agentsh.limits import (
    MAX_OUTPUT_BYTES,
    line_overrun_marker,
    truncate_text,
    truncation_marker,
)


async def read_until_sentinel(
    stdout: asyncio.StreamReader,
    sentinel_prefix: str,
    *,
    max_bytes: int = MAX_OUTPUT_BYTES,
    transform: Callable[[str], str] | None = None,
    strip_noise: Callable[[str], str] | None = None,
) -> tuple[str, str]:
    """Collect decoded stdout lines until one starts with sentinel_prefix.

    Args:
        stdout: The subprocess stdout stream.
        sentinel_prefix: Lines starting with this mark the end of output.
        max_bytes: Hard cap on buffered output before truncation kicks in.
        transform: Optional per-line post-decode transform (e.g. cmd.exe's
            CRLF normalisation) applied only to lines kept in the output.
        strip_noise: Optional per-line post-decode cleanup applied before
            *both* the sentinel-prefix check and transform/appending (e.g.
            PowerShell's `-Command -` mode unconditionally prefixing every
            line with VT100 escape codes). Unlike transform, this must run
            before the sentinel check too, since the noise would otherwise
            corrupt the prefix match, not just the kept output.

    Returns:
        tuple[str, str]: The collected output, and the raw decoded
        sentinel line (empty string if the stream ended first).
    """
    sentinel_bytes = sentinel_prefix.encode()
    chunks: list[str] = []
    marker: str | None = None
    total = 0

    def finish() -> str:
        # truncate_text is a safety net on the *kept* content only: the
        # marker is exempt so a small max_bytes (or decode inflation from
        # errors="replace") can never chew into the marker text itself.
        text = truncate_text("".join(chunks), max_bytes)
        return text + marker if marker else text

    while True:
        try:
            line = await stdout.readuntil(b"\n")
        except asyncio.IncompleteReadError as e:
            line = e.partial
        except asyncio.LimitOverrunError as e:
            if marker is None:
                marker = line_overrun_marker()
            await stdout.readexactly(e.consumed)
            continue
        if not line:
            return finish(), ""
        if marker is not None:
            # Already discarding output: skip the decode unless this line
            # might be the sentinel we're still watching for. The raw
            # check runs even when strip_noise is set: sentinel_prefix
            # never starts with the noise pattern, so a raw match still
            # means a real sentinel, and it avoids a decode+strip on
            # every discarded line for a caller whose noise doesn't
            # appear on every line.
            if line.startswith(sentinel_bytes):
                return finish(), line.decode(errors="replace")
            if strip_noise is not None:
                candidate = strip_noise(line.decode(errors="replace"))
                if candidate.startswith(sentinel_prefix):
                    return finish(), candidate
            continue
        decoded = line.decode(errors="replace")
        if strip_noise is not None:
            decoded = strip_noise(decoded)
        if decoded.startswith(sentinel_prefix):
            return finish(), decoded
        total += len(line)
        if total > max_bytes:
            marker = truncation_marker(max_bytes)
            continue
        chunks.append(transform(decoded) if transform else decoded)
