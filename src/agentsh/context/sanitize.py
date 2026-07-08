"""Boundary-wrapping and sanitization for untrusted context-provider output.

Context providers (git, docker, kubernetes, filesystem, shell history, ...)
surface strings that originate outside agentsh's control -- a git branch
name, a container name, a filesystem entry -- and those strings are
placed into the LLM system prompt. Without an explicit, unspoofable
boundary and sanitization, an attacker who controls one of those strings
(e.g. by naming a branch "IGNORE PREVIOUS INSTRUCTIONS...") can perform
indirect prompt injection, since the model has no way to distinguish
trusted instructions from untrusted environmental data.

Every fragment rendered into a system prompt must go through
`render_context_fragment`, which sanitizes the fragment's text and wraps
it in `CONTEXT_OPEN_TAG` / `CONTEXT_CLOSE_TAG` markers.
"""

import json

from agentsh.models import ContextFragment

CONTEXT_OPEN_TAG = "<untrusted-context>"
CONTEXT_CLOSE_TAG = "</untrusted-context>"

MAX_FRAGMENT_CHARS = 4000

_TRUNCATION_MARKER = "\n...[truncated at {limit} chars]...\n"


def sanitize_context_text(
    text: str, max_chars: int = MAX_FRAGMENT_CHARS
) -> str:
    """Neutralize boundary-spoofing markup and cap length of untrusted text.

    Escapes every `<` and `>` so no embedded content can render as a
    real tag -- including the boundary markers themselves -- regardless
    of case or whitespace variants. The result is then capped to
    `max_chars`, with a truncation marker appended if content was cut.
    """
    escaped = text.replace("<", "&lt;").replace(">", "&gt;")
    if len(escaped) <= max_chars:
        return escaped
    return escaped[:max_chars] + _TRUNCATION_MARKER.format(limit=max_chars)


def render_context_fragment(fragment: ContextFragment) -> str:
    """Render one fragment as a sanitized, boundary-wrapped prompt block.

    This is the single point where every context provider's output is
    funneled before it reaches an LLM system prompt, so no provider or
    backend needs to duplicate the sanitization logic.
    """
    payload_json = json.dumps(fragment.payload, indent=2)
    summary = sanitize_context_text(fragment.summary)
    payload = sanitize_context_text(payload_json)
    return (
        f"\n## {summary}\n"
        f"{CONTEXT_OPEN_TAG}\n"
        f"```json\n{payload}\n```\n"
        f"{CONTEXT_CLOSE_TAG}"
    )
