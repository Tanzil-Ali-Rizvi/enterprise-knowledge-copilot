"""Core data structures shared across the pipeline."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Document(BaseModel):
    doc_id: str
    title: str
    source: str  # relative path of the original file
    text: str
    department: str = "general"
    allowed_roles: list[str] = Field(default_factory=lambda: ["all"])
    updated: str | None = None


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    title: str
    source: str
    section: str
    text: str
    allowed_roles: list[str]
    position: int  # order within the document


class ScoredChunk(BaseModel):
    chunk: Chunk
    score: float
    retriever: str  # "bm25" | "vector" | "hybrid" | "rerank"


class Citation(BaseModel):
    index: int
    doc_id: str
    title: str
    section: str
    source: str
    snippet: str


class Answer(BaseModel):
    question: str
    answer: str
    citations: list[Citation]
    abstained: bool
    confidence: float
    timings_ms: dict[str, float]
    retrieved: list[str] = Field(default_factory=list, description="chunk ids in ranked order")
    usage: dict[str, int] = Field(default_factory=dict)
