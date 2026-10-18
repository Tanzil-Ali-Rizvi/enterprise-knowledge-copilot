from pathlib import Path

import pytest

from kcopilot.config import Settings
from kcopilot.pipeline import Copilot

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def settings() -> Settings:
    return Settings(
        data_dir=ROOT / "data" / "corpus",
        users_file=ROOT / "data" / "users.json",
        embedder="hashing",
        reranker="lexical",
        llm_provider="offline",
    )


@pytest.fixture(scope="session")
def copilot(settings) -> Copilot:
    return Copilot(settings=settings, persist=False)
