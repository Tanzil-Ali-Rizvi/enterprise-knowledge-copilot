"""Fusion (Reciprocal Rank Fusion) and reranking."""

from __future__ import annotations

import math
from typing import Protocol

from .models import ScoredChunk
from .text import tokenize


def reciprocal_rank_fusion(result_lists: list[list[ScoredChunk]], k: int = 60) -> list[ScoredChunk]:
    """Combine rankings without needing comparable scores: score = sum(1 / (k + rank))."""
    fused: dict[str, float] = {}
    by_id = {}
    for results in result_lists:
        for rank, item in enumerate(results, start=1):
            cid = item.chunk.chunk_id
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (k + rank)
            by_id[cid] = item.chunk
    ordered = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
    return [ScoredChunk(chunk=by_id[cid], score=s, retriever="hybrid") for cid, s in ordered]


class Reranker(Protocol):
    name: str

    def rerank(self, query: str, items: list[ScoredChunk], top_k: int) -> list[ScoredChunk]: ...


class NoReranker:
    """Passes items through; confidence is a normalised retrieval score."""

    name = "none"

    def rerank(self, query: str, items: list[ScoredChunk], top_k: int) -> list[ScoredChunk]:
        if not items:
            return []
        q = set(tokenize(query))
        out = []
        for it in items[:top_k]:
            words = set(tokenize(f"{it.chunk.title} {it.chunk.section} {it.chunk.text}"))
            coverage = len(q & words) / max(1, len(q))
            out.append(ScoredChunk(chunk=it.chunk, score=coverage, retriever="rerank"))
        return out


class LexicalReranker:
    """Offline reranker: IDF-weighted query-term coverage with a proximity bonus, blended with the
    fused retrieval rank. Score is in [0, 1] so it doubles as an answerability confidence."""

    name = "lexical"

    def __init__(self, doc_freq: dict[str, int] | None = None, n_docs: int = 1):
        self.doc_freq = doc_freq or {}
        self.n_docs = max(1, n_docs)

    def _idf(self, term: str) -> float:
        df = self.doc_freq.get(term, 0)
        return math.log(1 + (self.n_docs - df + 0.5) / (df + 0.5))

    def rerank(self, query: str, items: list[ScoredChunk], top_k: int) -> list[ScoredChunk]:
        q_terms = list(dict.fromkeys(tokenize(query)))
        if not q_terms or not items:
            return NoReranker().rerank(query, items, top_k)
        total_idf = sum(self._idf(t) for t in q_terms) or 1.0
        scored = []
        for rank, it in enumerate(items, start=1):
            body = tokenize(f"{it.chunk.title} {it.chunk.section} {it.chunk.text}")
            words = set(body)
            prefixes = {w[:5] for w in words if len(w) > 5}
            hit = 0.0
            for t in q_terms:
                if t in words:
                    hit += self._idf(t)
                elif len(t) > 5 and t[:5] in prefixes:
                    hit += 0.6 * self._idf(t)
            coverage = hit / total_idf
            # bigram proximity bonus
            pairs = set(zip(body, body[1:], strict=False))
            q_pairs = list(zip(q_terms, q_terms[1:], strict=False))
            prox = sum(1 for p in q_pairs if p in pairs) / max(1, len(q_pairs))
            rank_prior = 1.0 / (1 + 0.15 * (rank - 1))
            score = 0.65 * coverage + 0.15 * prox + 0.20 * rank_prior
            scored.append(ScoredChunk(chunk=it.chunk, score=round(score, 4), retriever="rerank"))
        scored.sort(key=lambda s: s.score, reverse=True)
        return scored[:top_k]


class CrossEncoderReranker:
    """Neural cross-encoder (ms-marco MiniLM) via fastembed; logits squashed to [0, 1]."""

    name = "cross-encoder"

    def __init__(self, model: str = "Xenova/ms-marco-MiniLM-L-6-v2"):
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self._model = TextCrossEncoder(model_name=model)

    def rerank(self, query: str, items: list[ScoredChunk], top_k: int) -> list[ScoredChunk]:
        if not items:
            return []
        docs = [f"{it.chunk.title}. {it.chunk.section}. {it.chunk.text}" for it in items]
        logits = list(self._model.rerank(query, docs))
        scored = [
            ScoredChunk(chunk=it.chunk, score=round(1 / (1 + math.exp(-float(lg))), 4), retriever="rerank")
            for it, lg in zip(items, logits, strict=True)
        ]
        scored.sort(key=lambda s: s.score, reverse=True)
        return scored[:top_k]
