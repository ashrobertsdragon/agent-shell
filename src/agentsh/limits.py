"""Shared output-size limits for shell backends and file/command tools.

A single unbounded read -- a command emitting hundreds of megabytes, or a
file of the same size -- must never be buffered whole in memory or shipped
whole into an LLM prompt. Every read path that touches external output
(subprocess stdout/stderr, file contents) enforces MAX_OUTPUT_BYTES via
the helpers below.
"""

import os
from pathlib import Path

MAX_OUTPUT_BYTES = 1024 * 1024


def truncation_marker(max_bytes: int = MAX_OUTPUT_BYTES) -> str:
    """Return the marker appended when output is truncated at max_bytes."""
    return f"\n... [output truncated at {max_bytes} bytes] ...\n"


def line_overrun_marker() -> str:
    """Return the marker for a single line exceeding the stream's own limit.

    Distinct from truncation_marker(): this case is capped by asyncio's
    internal per-line buffer limit, not by MAX_OUTPUT_BYTES, so claiming
    the byte-count cap here would misstate the actual cause.
    """
    return (
        "\n... [output truncated: a single line exceeded the internal "
        "read buffer] ...\n"
    )


def truncate_text(text: str, max_bytes: int = MAX_OUTPUT_BYTES) -> str:
    """Truncate text to at most max_bytes of UTF-8, appending a marker.

    Returns text unchanged when it already fits within max_bytes. Used as
    a defense-in-depth cap on already-collected strings; readers that can
    stream their source (subprocess stdout, files) should stop reading at
    the cap instead of relying solely on this.
    """
    encoded = text.encode("utf-8", errors="replace")
    if len(encoded) <= max_bytes:
        return text
    truncated = encoded[:max_bytes].decode("utf-8", errors="ignore")
    return truncated + truncation_marker(max_bytes)


def read_capped_text(
    path: str | os.PathLike[str],
    max_bytes: int = MAX_OUTPUT_BYTES,
    errors: str = "replace",
) -> str:
    """Read at most max_bytes from path, appending a marker if truncated.

    Reads raw bytes and stops at the cap so a huge file is never loaded
    whole into memory just to be thrown away afterwards.
    """
    with Path(path).open("rb") as f:
        data = f.read(max_bytes + 1)
    if len(data) <= max_bytes:
        return data.decode("utf-8", errors=errors)
    return data[:max_bytes].decode("utf-8", errors=errors) + truncation_marker(
        max_bytes
    )
