"""Bounded, crash-safe readline loop shared by the persistent shell backends.

asyncio.StreamReader.readline() raises ValueError (LimitOverrunError under
the hood) when a single line exceeds its internal buffer limit -- a
command that emits one very long line (binary data, minified assets, a
huge log line) would otherwise crash the whole process. This helper
swallows that error, records a truncation marker, and keeps reading so
the sentinel line further down the stream is still reached, keeping the
shell's request/response protocol in sync for the next command. It also
caps total buffered output at MAX_OUTPUT_BYTES so a command that emits
gigabytes of ordinary output cannot exhaust memory or blow up an LLM
prompt.
"""

import asyncio
from collections.abc import Callable

from agentsh.limits import MAX_OUTPUT_BYTES, truncation_marker


async def read_until_sentinel(
    stdout: asyncio.StreamReader,
    sentinel_prefix: str,
    *,
    max_bytes: int = MAX_OUTPUT_BYTES,
    transform: Callable[[str], str] | None = None,
) -> tuple[str, str]:
    """Collect decoded stdout lines until one starts with sentinel_prefix.

    Args:
        stdout: The subprocess stdout stream.
        sentinel_prefix: Lines starting with this mark the end of output.
        max_bytes: Hard cap on buffered output before truncation kicks in.
        transform: Optional per-line post-decode transform (e.g. cmd.exe's
            CRLF normalisation) applied only to lines kept in the output.

    Returns:
        tuple[str, str]: The collected output, and the raw decoded
        sentinel line (empty string if the stream ended first).
    """
    chunks: list[str] = []
    total = 0
    truncated = False
    while True:
        try:
            line = await stdout.readline()
        except ValueError:
            if not truncated:
                chunks.append(truncation_marker(max_bytes))
                truncated = True
            continue
        if not line:
            return "".join(chunks), ""
        decoded = line.decode(errors="replace")
        if decoded.startswith(sentinel_prefix):
            return "".join(chunks), decoded
        if truncated:
            continue
        total += len(line)
        if total > max_bytes:
            chunks.append(truncation_marker(max_bytes))
            truncated = True
            continue
        chunks.append(transform(decoded) if transform else decoded)
