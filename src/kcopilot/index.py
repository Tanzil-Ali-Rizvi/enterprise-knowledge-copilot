"""Hybrid index: Qdrant (dense vectors) + BM25 (keywords), both access-filtered.

By default Qdrant runs in-memory inside each process, and the chunk text + embeddings are cached in
KC_INDEX_DIR so restarts don't re-embed. Many processes (API, UI, CLI) can share that cache safely.
Set KC_QDRANT_URL to use a Qdrant server / Qdrant Cloud cluster instead for large corpora.
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, FieldCondition, Filter, MatchAny, PointStruct, VectorParams
from rank_bm25 import BM25Okapi

from .embeddings import Embedder
from .models import Chunk, ScoredChunk
from .text import tokenize

CACHE_FILE = "index.json"


def embed_text(chunk: Chunk) -> str:
    """Prepend title and section so short chunks keep their context."""
    header = f"{chunk.title}. {chunk.section}." if chunk.section else f"{chunk.title}."
    return f"{header} {chunk.text}"


class HybridIndex:
    def __init__(
        self,
        embedder: Embedder,
        collection: str = "knowledge",
        path: Path | None = None,
        qdrant_url: str | None = None,
    ):
        self.embedder = embedder
        self.collection = collection
        self.path = Path(path) if path else None
        self.client = QdrantClient(url=qdrant_url) if qdrant_url else QdrantClient(":memory:")
        self.chunks: dict[str, Chunk] = {}
        self._order: list[str] = []
        self._bm25: BM25Okapi | None = None

    # ---------- building ----------
    def build(self, chunks: list[Chunk]) -> None:
        vectors = self.embedder.embed_documents([embed_text(c) for c in chunks])
        self._upload(chunks, vectors)
        if self.path:
            self.path.mkdir(parents=True, exist_ok=True)
            payload = {
                "embedder": self.embedder.name,
                "chunks": [c.model_dump() for c in chunks],
                "vectors": [[round(x, 6) for x in v] for v in vectors],
            }
            tmp = self.path / f".{CACHE_FILE}.{os.getpid()}.tmp"
            tmp.write_text(json.dumps(payload), encoding="utf-8")
            tmp.replace(self.path / CACHE_FILE)  # atomic: concurrent readers never see a partial file

    def load(self) -> bool:
        """Load the cached index. Returns False if missing or built with a different embedder."""
        cache = self.path / CACHE_FILE if self.path else None
        if not cache or not cache.exists():
            return False
        data = json.loads(cache.read_text(encoding="utf-8"))
        if data.get("embedder") != self.embedder.name:
            return False
        self._upload([Chunk(**c) for c in data["chunks"]], data["vectors"])
        return True

    def _upload(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)
        self.client.create_collection(
            self.collection, vectors_config=VectorParams(size=self.embedder.dim, distance=Distance.COSINE)
        )
        points = [
            PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, c.chunk_id)),
                vector=v,
                payload={"chunk_id": c.chunk_id, "allowed_roles": c.allowed_roles},
            )
            for c, v in zip(chunks, vectors, strict=True)
        ]
        for i in range(0, len(points), 256):
            self.client.upsert(self.collection, points=points[i : i + 256])
        self._set_chunks(chunks)

    def _set_chunks(self, chunks: list[Chunk]) -> None:
        self.chunks = {c.chunk_id: c for c in chunks}
        self._order = [c.chunk_id for c in chunks]
        self._bm25 = BM25Okapi([tokenize(embed_text(c)) or ["_"] for c in chunks]) if chunks else None

    def __len__(self) -> int:
        return len(self.chunks)

    # ---------- searching ----------
    def vector_search(self, query: str, roles: list[str], limit: int) -> list[ScoredChunk]:
        res = self.client.query_points(
            self.collection,
            query=self.embedder.embed_query(query),
            query_filter=Filter(must=[FieldCondition(key="allowed_roles", match=MatchAny(any=roles))]),
            limit=limit,
        )
        return [
            ScoredChunk(chunk=self.chunks[p.payload["chunk_id"]], score=float(p.score), retriever="vector")
            for p in res.points
        ]

    def bm25_search(self, query: str, roles: list[str], limit: int) -> list[ScoredChunk]:
        if not self._bm25:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        allowed = set(roles)
        ranked = sorted(
            (
                (s, cid)
                for s, cid in zip(scores, self._order, strict=True)
                if s > 0 and allowed & set(self.chunks[cid].allowed_roles)
            ),
            reverse=True,
        )[:limit]
        return [ScoredChunk(chunk=self.chunks[cid], score=float(s), retriever="bm25") for s, cid in ranked]
