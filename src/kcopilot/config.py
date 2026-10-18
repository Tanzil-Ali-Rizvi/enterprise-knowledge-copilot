"""Central configuration, loaded from environment variables / .env (prefix KC_)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="KC_", env_file=".env", extra="ignore")

    # Data
    data_dir: Path = Path("data/corpus")
    users_file: Path = Path("data/users.json")
    index_dir: Path = Path(".index")
    collection: str = "knowledge"
    qdrant_url: str | None = None  # e.g. http://localhost:6333 or a Qdrant Cloud URL

    # Chunking
    chunk_size_words: int = 180
    chunk_overlap_words: int = 40

    # Retrieval
    embedder: Literal["hashing", "fastembed"] = "hashing"
    embed_model: str = "BAAI/bge-small-en-v1.5"
    reranker: Literal["none", "lexical", "cross-encoder"] = "lexical"
    rerank_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    retrieval_mode: Literal["hybrid", "vector", "bm25"] = "hybrid"
    candidates: int = 20  # per retriever, before fusion
    top_k: int = 5
    rrf_k: int = 60
    min_confidence: float = 0.40

    # Generation
    llm_provider: Literal["offline", "openai"] = "offline"
    llm_base_url: str = "http://localhost:11434/v1"
    llm_model: str = "llama3.2:3b"
    llm_api_key: str = "not-needed"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 600
    llm_timeout_s: float = 60.0


@lru_cache
def get_settings() -> Settings:
    return Settings()
