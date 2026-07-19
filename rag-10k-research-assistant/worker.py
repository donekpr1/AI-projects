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
    run_auto_with_fallback,
    cache_metrics,
)

# ── CHANGE 1 — import run_agentic from agent.py ──────────────
# agent.py imports from pipeline.py internally
# worker.py only needs to import the entry point
from agent import run_agentic


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
    pipeline: str = "Auto (Adaptive + fallback)"
    session_id: str = "default"
    chat_history: Optional[List[ChatMessage]] = None


class CompareRequest(BaseModel):
    query: str
    session_id: str = "default"
    chat_history: Optional[List[ChatMessage]] = None
    # ── CHANGE 2 — added pipelines field ─────────────────────
    # app.py now sends which pipelines to compare
    # defaults to Vector + Vectorless if not specified
    # Optional so old app.py versions still work
    pipelines: Optional[List[str]] = None


# ── CHANGE 3 — new request model for agentic endpoint ────────
# Separate model from AskRequest because agentic never needs
# the pipeline field — it always runs agentic logic
class AgenticRequest(BaseModel):
    query: str
    session_id: str = "default"
    chat_history: Optional[List[ChatMessage]] = None


def _history_dicts(req_history: Optional[List[ChatMessage]]):
    if not req_history:
        return None
    return [{"role": m.role, "content": m.content} for m in req_history]


def _run_pipeline(name: str, query: str, **kwargs) -> dict:
    """
    Helper that maps pipeline name string to the right function.
    Used by both /ask and /compare endpoints.
    """
    if name == "Vector RAG":
        return run_vector(query, **kwargs)
    if name == "Vectorless RAG":
        return run_vectorless(query, **kwargs)
    if name == "Agentic RAG":
        return run_agentic(query, **kwargs)
    if name in (
        "Auto (Adaptive + fallback)",
        "Auto",
        "Adaptive + fallback",
    ):
        return run_auto_with_fallback(query, **kwargs)
    # Default — Adaptive RAG only (no fallback)
    return run_adaptive(query, **kwargs)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/cache_stats")
def cache_stats():
    total = cache_metrics.get("total_queries", 0) or 0
    hits = (
        cache_metrics.get("stage1_hits", 0) +
        cache_metrics.get("stage2_hits", 0)
    )
    return {
        **cache_metrics,
        "hit_rate": round(hits / total, 3) if total else 0.0,
    }


@app.post("/ask")
def ask(req: AskRequest):
    """
    Standard endpoint for all pipelines except Agentic RAG.
    Agentic RAG has its own /agentic endpoint.
    If somehow "Agentic RAG" is sent here, _run_pipeline
    handles it correctly anyway via the helper.
    """
    history = _history_dicts(req.chat_history)
    kwargs = {"session_id": req.session_id, "chat_history": history}
    return _run_pipeline(req.pipeline, req.query, **kwargs)


# ── CHANGE 4 — new /agentic endpoint ─────────────────────────
# app.py routes here when user selects "Agentic RAG"
# Calls run_agentic() from agent.py
# agent.py imports retrieval functions from pipeline.py internally
# Returns same dict format as other pipelines so app.py works unchanged
@app.post("/agentic")
def agentic(req: AgenticRequest):
    """
    Dedicated endpoint for Agentic RAG.
    Triggers the ReAct loop in agent.py.

    Why separate from /ask:
      agent.py is a different module from pipeline.py
      Keeping it separate makes the routing explicit
      and avoids adding agentic logic to the pipeline module

    Flow:
      app.py → POST /agentic
      worker.py → run_agentic() in agent.py
      agent.py → cache check → episodic recall → ReAct loop
      ReAct loop → calls pipeline.py retrieval functions as tools
      → returns result dict → worker returns JSON → app.py displays
    """
    history = _history_dicts(req.chat_history)
    return run_agentic(
        query=req.query,
        session_id=req.session_id,
        chat_history=history,
    )


# ── CHANGE 5 — updated /compare to support any pipelines ─────
# Old version hardcoded Vector vs Vectorless
# New version accepts a list of pipeline names from app.py
# Returns dict keyed by pipeline name instead of hardcoded keys
@app.post("/compare")
def compare(req: CompareRequest):
    """
    Runs multiple pipelines on the same query and returns all results.
    app.py displays them side by side.

    Old behavior (pipelines not specified):
      Returns {"vector": ..., "vectorless": ...}
      Backward compatible — old app.py still works

    New behavior (pipelines list specified):
      Returns {"Vector RAG": ..., "Adaptive RAG": ..., ...}
      Keyed by exact pipeline name for flexible comparison

    Runs pipelines sequentially — if user compares 3 pipelines,
    all 3 run one after another. For production with many users,
    these could be parallelised with ThreadPoolExecutor.
    """
    history = _history_dicts(req.chat_history)
    kwargs = {"session_id": req.session_id, "chat_history": history}

    # New behavior — pipelines list provided by app.py
    if req.pipelines:
        results = {}
        for pipe_name in req.pipelines:
            results[pipe_name] = _run_pipeline(pipe_name, req.query, **kwargs)
        return results

    # Old behavior — backward compatible with original app.py
    # Returns hardcoded vector/vectorless keys
    return {
        "vector":     run_vector(req.query, **kwargs),
        "vectorless": run_vectorless(req.query, **kwargs),
    }