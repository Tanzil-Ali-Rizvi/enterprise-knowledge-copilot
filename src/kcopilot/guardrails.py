"""Ingestion-time prompt-injection guardrail.

Retrieval-time ACLs already guarantee that injected text cannot unlock restricted documents. This
layer additionally removes instruction-like sentences from documents before indexing, so they are
never shown to the LLM (or to users) at all, and logs them for review.

It is a heuristic first line of defence, not a complete solution; the system prompt still treats all
sources as untrusted data.
"""

from __future__ import annotations

import logging
import re

log = logging.getLogger("kcopilot.guardrails")

_PATTERNS = [
    r"ignore (all |any )?(previous|prior|above|earlier) (instructions|rules|prompts)",
    r"disregard (all |any )?(previous|prior|above|the) (instructions|rules|system prompt)",
    r"you are now (in )?\w+ mode",
    r"(enter|enable|activate) (admin|developer|god|jailbreak) mode",
    r"reveal (the |your )?(system prompt|hidden instructions)",
    r"(new|updated) system (prompt|instructions)\s*:",
]
_INJECTION = re.compile("|".join(_PATTERNS), re.IGNORECASE)
_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")


def looks_like_injection(text: str) -> bool:
    return bool(_INJECTION.search(text))


def sanitize(text: str, source: str = "") -> tuple[str, int]:
    """Drop sentences that look like prompt injection. Returns (clean text, number removed)."""
    if not looks_like_injection(text):
        return text, 0
    kept, removed = [], 0
    for line in text.split("\n"):
        parts = _SENTENCE.split(line)
        good = [p for p in parts if not looks_like_injection(p)]
        removed += len(parts) - len(good)
        kept.append(" ".join(good))
    if removed:
        log.warning("quarantined %d injection-like sentence(s) in %s", removed, source or "<text>")
    return "\n".join(kept), removed
