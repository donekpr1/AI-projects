"""Local RAG worker — keeps models warm. Start once, leave running."""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel

from pipelines import init_pipelines, run_adaptive, run_vector, run_vectorless


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs once when worker starts — this is the slow part
    print("Loading models & indexes...")
    init_pipelines()
    print("Worker ready.")
    yield


app = FastAPI(title="RAG Worker", lifespan=lifespan)


class AskRequest(BaseModel):
    query: str
    pipeline: str = "Adaptive RAG"  # Adaptive RAG | Vector RAG | Vectorless RAG


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask")
def ask(req: AskRequest):
    if req.pipeline == "Vector RAG":
        return run_vector(req.query)
    if req.pipeline == "Vectorless RAG":
        return run_vectorless(req.query)
    return run_adaptive(req.query)


@app.post("/compare")
def compare(req: AskRequest):
    return {
        "vector": run_vector(req.query),
        "vectorless": run_vectorless(req.query),
    }