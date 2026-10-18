"""FastAPI service.

Run:  uvicorn kcopilot.api:app --reload
Docs: http://localhost:8000/docs

Auth note: for the demo the caller passes `X-User-Id`. In production this comes from a verified
JWT/OIDC token issued by your identity provider; never trust a client-supplied header directly.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from . import __version__
from .models import Answer
from .pipeline import Copilot
from .security import UnknownUserError, User

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

_state: dict[str, Copilot] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    _state["copilot"] = Copilot()
    yield
    _state.clear()


app = FastAPI(title="Enterprise Knowledge Copilot", version=__version__, lifespan=lifespan)


def copilot() -> Copilot:
    return _state["copilot"]


def current_user(
    x_user_id: str = Header(..., description="Demo identity header"), cp: Copilot = Depends(copilot)
) -> User:
    try:
        return cp.users.get(x_user_id)
    except UnknownUserError:
        raise HTTPException(status_code=401, detail="Unknown user") from None


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    top_k: int | None = Field(None, ge=1, le=20)
    mode: str | None = Field(None, pattern="^(hybrid|vector|bm25)$")


@app.get("/health")
def health(cp: Copilot = Depends(copilot)) -> dict:
    return {
        "status": "ok",
        "version": __version__,
        "chunks": len(cp.index),
        "embedder": cp.embedder.name,
        "reranker": cp.reranker.name,
        "llm": cp.llm.name,
    }


@app.post("/ask", response_model=Answer)
def ask(req: AskRequest, user: User = Depends(current_user), cp: Copilot = Depends(copilot)) -> Answer:
    return cp.ask(req.question, user.user_id, mode=req.mode, top_k=req.top_k)


@app.get("/documents")
def documents(user: User = Depends(current_user), cp: Copilot = Depends(copilot)) -> list[dict]:
    """Only the documents this user is allowed to see."""
    return cp.visible_documents(user)


@app.post("/admin/reindex")
def reindex(user: User = Depends(current_user), cp: Copilot = Depends(copilot)) -> dict:
    if "admin" not in user.roles:
        raise HTTPException(status_code=403, detail="Admin role required")
    return cp.ingest(cp.settings.data_dir)
