"""Embedding backends.

- HashingEmbedder: deterministic, zero-download feature hashing (unigrams + bigrams). Lets the whole
  system, tests and CI run fully offline. Weaker semantically; use for demos/CI.
- FastEmbedEmbedder: BAAI/bge-small-en-v1.5 via ONNX (fastembed). Recommended for real use.
"""

from __future__ import annotations

import hashlib
import math
from typing import Protocol

from .text import tokenize


class Embedder(Protocol):
    name: str
    dim: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


class HashingEmbedder:
    def __init__(self, dim: int = 1024):
        self.dim = dim
        self.name = f"hashing-{dim}"

    def _vec(self, text: str) -> list[float]:
        toks = tokenize(text)
        feats = toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:], strict=False)]
        # light prefix stemming so "reimburse"/"reimbursement" share a bucket
        feats += [t[:5] for t in toks if len(t) > 5]
        vec = [0.0] * self.dim
        for f in feats:
            h = int.from_bytes(hashlib.md5(f.encode()).digest()[:8], "little")
            vec[h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


class FastEmbedEmbedder:
    def __init__(self, model: str = "BAAI/bge-small-en-v1.5"):
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model_name=model)
        self.name = model
        self.dim = len(next(iter(self._model.embed(["probe"]))))

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in self._model.passage_embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self._model.query_embed(text))).tolist()


def build_embedder(kind: str, model: str) -> Embedder:
    if kind == "fastembed":
        return FastEmbedEmbedder(model)
    return HashingEmbedder()
