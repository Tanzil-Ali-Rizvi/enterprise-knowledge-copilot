"""The Copilot: retrieve -> fuse -> rerank -> (abstain | generate) -> cite."""

from __future__ import annotations

import json
import logging
import time
from collections import Counter
from pathlib import Path

from .config import Settings, get_settings
from .embeddings import build_embedder
from .generation import SYSTEM_PROMPT, build_user_prompt, extract_citations, format_sources, is_abstention
from .index import HybridIndex, embed_text
from .ingest import chunk_document, load_directory
from .llm import ABSTAIN_MESSAGE, LLM, build_llm
from .models import Answer, ScoredChunk
from .retrieval import CrossEncoderReranker, LexicalReranker, NoReranker, Reranker, reciprocal_rank_fusion
from .security import User, UserDirectory, can_read
from .text import tokenize

log = logging.getLogger("kcopilot")


class Copilot:
    def __init__(
        self,
        settings: Settings | None = None,
        llm: LLM | None = None,
        users: UserDirectory | None = None,
        persist: bool = True,
    ):
        self.settings = s = settings or get_settings()
        self.embedder = build_embedder(s.embedder, s.embed_model)
        self.index = HybridIndex(
            self.embedder, s.collection, s.index_dir if persist else None, s.qdrant_url if persist else None
        )
        self.llm = llm or build_llm(s)
        self.users = users or UserDirectory.from_file(s.users_file)
        self.reranker: Reranker = NoReranker()
        self.documents: dict[str, dict] = {}
        if not self.index.load():
            self.ingest(s.data_dir)
        else:
            self._after_index()

    # ---------- ingestion ----------
    def ingest(self, data_dir: Path | str) -> dict:
        t0 = time.perf_counter()
        docs = load_directory(Path(data_dir))
        chunks = [
            c
            for d in docs
            for c in chunk_document(d, self.settings.chunk_size_words, self.settings.chunk_overlap_words)
        ]
        self.index.build(chunks)
        self._after_index()
        stats = {
            "documents": len(docs),
            "chunks": len(chunks),
            "seconds": round(time.perf_counter() - t0, 2),
            "embedder": self.embedder.name,
        }
        log.info("ingested %s", stats)
        return stats

    def _after_index(self) -> None:
        df: Counter[str] = Counter()
        self.documents = {}
        for c in self.index.chunks.values():
            df.update(set(tokenize(embed_text(c))))
            self.documents.setdefault(
                c.doc_id,
                {
                    "doc_id": c.doc_id,
                    "title": c.title,
                    "source": c.source,
                    "allowed_roles": c.allowed_roles,
                    "chunks": 0,
                },
            )
            self.documents[c.doc_id]["chunks"] += 1
        kind = self.settings.reranker
        if kind == "cross-encoder":
            self.reranker = CrossEncoderReranker(self.settings.rerank_model)
        elif kind == "lexical":
            self.reranker = LexicalReranker(dict(df), len(self.index))
        else:
            self.reranker = NoReranker()

    # ---------- querying ----------
    def visible_documents(self, user: User) -> list[dict]:
        return [d for d in self.documents.values() if can_read(user, d["allowed_roles"])]

    def retrieve(
        self, question: str, user: User, mode: str | None = None, top_k: int | None = None
    ) -> tuple[list[ScoredChunk], dict[str, float]]:
        s = self.settings
        mode = mode or s.retrieval_mode
        top_k = top_k or s.top_k
        roles = user.effective_roles
        timings: dict[str, float] = {}

        t = time.perf_counter()
        lists = []
        if mode in ("hybrid", "vector"):
            lists.append(self.index.vector_search(question, roles, s.candidates))
        if mode in ("hybrid", "bm25"):
            lists.append(self.index.bm25_search(question, roles, s.candidates))
        fused = reciprocal_rank_fusion(lists, s.rrf_k) if len(lists) > 1 else (lists[0] if lists else [])
        timings["retrieve_ms"] = _ms(t)

        t = time.perf_counter()
        ranked = self.reranker.rerank(question, fused, top_k)
        timings["rerank_ms"] = _ms(t)

        # Defence in depth: re-check ACL on the final set even though both retrievers already filter.
        ranked = [r for r in ranked if can_read(user, r.chunk.allowed_roles)]
        return ranked, timings

    def ask(self, question: str, user_id: str, mode: str | None = None, top_k: int | None = None) -> Answer:
        t_total = time.perf_counter()
        user = self.users.get(user_id)
        items, timings = self.retrieve(question, user, mode, top_k)
        confidence = items[0].score if items else 0.0

        usage: dict[str, int] = {}
        if not items or confidence < self.settings.min_confidence:
            # Cheap, safe path: no LLM call when retrieval found nothing convincing.
            text, citations, abstained = ABSTAIN_MESSAGE, [], True
            timings["generate_ms"] = 0.0
        else:
            t = time.perf_counter()
            result = self.llm.generate(
                SYSTEM_PROMPT, build_user_prompt(question, items), format_sources(items), question
            )
            timings["generate_ms"] = _ms(t)
            usage = result.usage
            text, citations = extract_citations(result.text, items)
            abstained = is_abstention(text)
            if abstained:
                citations = []

        timings["total_ms"] = _ms(t_total)
        answer = Answer(
            question=question,
            answer=text,
            citations=citations,
            abstained=abstained,
            confidence=round(confidence, 4),
            timings_ms=timings,
            retrieved=[i.chunk.chunk_id for i in items],
            usage=usage,
        )
        log.info(
            json.dumps(
                {
                    "event": "ask",
                    "user": user_id,
                    "roles": user.roles,
                    "abstained": abstained,
                    "confidence": answer.confidence,
                    "n_citations": len(citations),
                    **timings,
                }
            )
        )
        return answer


def _ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000, 2)
