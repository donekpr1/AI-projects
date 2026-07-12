"""Local RAG worker — keeps models/indexes warm; runs cache + episodic + pipelines."""
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI
from pydantic import BaseModel

from pipelines import (
    init_pipelines,
    run_adaptive,
    run_vector,
    run_vectorless,
    cache_metrics,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading models, disk Qdrant, corpus_index, cache, episodic...")
    init_pipelines()
    print("Worker ready.")
    yield


app = FastAPI(title="RAG Worker", lifespan=lifespan)


class ChatMessage(BaseModel):
    role: str
    content: str


class AskRequest(BaseModel):
    query: str
    pipeline: str = "Adaptive RAG"
    session_id: str = "default"
    chat_history: Optional[List[ChatMessage]] = None


class CompareRequest(BaseModel):
    query: str
    session_id: str = "default"
    chat_history: Optional[List[ChatMessage]] = None


def _history_dicts(req_history: Optional[List[ChatMessage]]):
    if not req_history:
        return None
    return [{"role": m.role, "content": m.content} for m in req_history]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/cache_stats")
def cache_stats():
    total = cache_metrics.get("total_queries", 0) or 0
    hits = cache_metrics.get("stage1_hits", 0) + cache_metrics.get("stage2_hits", 0)
    return {
        **cache_metrics,
        "hit_rate": round(hits / total, 3) if total else 0.0,
    }


@app.post("/ask")
def ask(req: AskRequest):
    history = _history_dicts(req.chat_history)
    kwargs = {"session_id": req.session_id, "chat_history": history}

    if req.pipeline == "Vector RAG":
        return run_vector(req.query, **kwargs)
    if req.pipeline == "Vectorless RAG":
        return run_vectorless(req.query, **kwargs)
    return run_adaptive(req.query, **kwargs)


@app.post("/compare")
def compare(req: CompareRequest):
    history = _history_dicts(req.chat_history)
    kwargs = {"session_id": req.session_id, "chat_history": history}
    return {
        "vector": run_vector(req.query, **kwargs),
        "vectorless": run_vectorless(req.query, **kwargs),
    }