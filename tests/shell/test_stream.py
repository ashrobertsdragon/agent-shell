"""Tests for the shared bounded readline loop used by shell plugins."""

import asyncio

from agentsh.limits import truncation_marker
from agentsh.shell.plugin._stream import read_until_sentinel


def _reader(*chunks: bytes, limit: int | None = None) -> asyncio.StreamReader:
    """Build a StreamReader pre-fed with chunks and already at EOF."""
    reader = (
        asyncio.StreamReader(limit=limit)
        if limit is not None
        else asyncio.StreamReader()
    )
    for chunk in chunks:
        reader.feed_data(chunk)
    reader.feed_eof()
    return reader


async def test_collects_lines_until_sentinel() -> None:
    """Lines before the sentinel are collected; the sentinel is returned."""
    reader = _reader(b"hello\n", b"world\n", b"SENTINEL:0:/tmp\n", b"ignored\n")
    output, sentinel = await read_until_sentinel(reader, "SENTINEL:")
    assert output == "hello\nworld\n"
    assert sentinel == "SENTINEL:0:/tmp\n"


async def test_returns_empty_sentinel_on_eof_without_sentinel() -> None:
    """A stream that ends before the sentinel yields an empty sentinel line."""
    reader = _reader(b"hello\n")
    output, sentinel = await read_until_sentinel(reader, "SENTINEL:")
    assert output == "hello\n"
    assert sentinel == ""


async def test_truncates_output_over_max_bytes() -> None:
    """Output beyond max_bytes is replaced by a single truncation marker."""
    reader = _reader(b"a" * 100 + b"\n", b"more\n", b"SENTINEL:0:/tmp\n")
    output, sentinel = await read_until_sentinel(
        reader, "SENTINEL:", max_bytes=10
    )
    assert output == truncation_marker(10)
    assert sentinel == "SENTINEL:0:/tmp\n"


async def test_swallows_oversized_line_value_error() -> None:
    """A single line beyond asyncio's own buffer limit doesn't raise.

    asyncio.StreamReader.readline() raises ValueError (LimitOverrunError)
    for a line longer than its internal limit; that must be swallowed so
    the sentinel further down the stream is still reached.
    """
    reader = _reader(b"x" * 200 + b"\n", b"SENTINEL:0:/tmp\n", limit=64)
    output, sentinel = await read_until_sentinel(reader, "SENTINEL:")
    assert "output truncated" in output
    assert sentinel == "SENTINEL:0:/tmp\n"


async def test_transform_applied_only_to_kept_lines() -> None:
    """The optional transform runs on lines kept in the output."""
    reader = _reader(b"a\r\n", b"SENTINEL:0:/tmp\n")
    output, sentinel = await read_until_sentinel(
        reader,
        "SENTINEL:",
        transform=lambda decoded: decoded.replace("\r\n", "\n"),
    )
    assert output == "a\n"
    assert sentinel == "SENTINEL:0:/tmp\n"
