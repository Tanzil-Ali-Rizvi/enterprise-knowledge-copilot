"""Small text utilities shared by BM25, the hashing embedder and the lexical reranker."""

from __future__ import annotations

import re

_TOKEN = re.compile(r"[a-z0-9]+(?:[.'][a-z0-9]+)*")

STOPWORDS = frozenset(
    """a an and are as at be by can do does for from has have how i in is it its may me my of on or our
    should so than that the their there these this to was we were what when where which who why will with
    you your am any all about into if not no can't don't""".split()
)


_HYPHENATED = re.compile(r"\b([a-z0-9]+)-([a-z0-9]+)\b")


def tokenize(text: str, drop_stopwords: bool = True) -> list[str]:
    # "Wi-Fi" -> "wifi wi fi" so both spellings match
    text = _HYPHENATED.sub(r"\1\2 \1 \2", text.lower())
    tokens = _TOKEN.findall(text)
    return [t for t in tokens if t not in STOPWORDS] if drop_stopwords else tokens


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])|\s+(?=- )|\s+(?=\d+\.\s)", text)
    return [p.strip(" -") for p in parts if len(p.strip()) > 3]
