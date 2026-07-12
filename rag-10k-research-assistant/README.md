RAG OPTIMIZATION PROJECT — SUMMARY
====================================
10-K Filing Research Assistant — EDGAR-CORPUS (2020), 8 companies

This file documents the full build process: what was tried, what worked,
what didn't, and why — based on a fixed 9-question eval set tested against
every pipeline change.

Shipped demo stack (current):
  Vector RAG · Vectorless RAG · Adaptive RAG · Agentic RAG (ReAct)
  Streamlit UI · FastAPI worker · disk Qdrant · semantic cache · episodic memory


PROJECT SETUP
-------------
Dataset:        eloukas/edgar-corpus, year_2020 (Hugging Face)
Scope:          8 companies (Microsoft, Alphabet, Amazon, Meta, Tesla,
                 Nvidia, Apple, JPMorgan Chase), section_1A (Risk Factors)
                Note: Amazon section_1A is empty in this slice (known gap).
Eval set:       9 hand-written questions with known expected answers
                 (5 factual, 2 comparative, 2 negative/out-of-scope)
Generation LLM: gpt-4o-mini (OpenAI)
Embedding:      BAAI/bge-base-en-v1.5 (local)
Reranker:       BAAI/bge-reranker-v2-m3 (local, CrossEncoder)
Vector store:   Qdrant on disk (qdrant_data/) — rebuilt only if missing
Vectorless:     corpus_index.json (structural nodes) + LLM TOC navigation
Cache:          raw + resolved semantic cache (cache_store.json)
Memory:         episodic Q&A in Qdrant collection episodic_memory
Demo:           Streamlit (app.py) + FastAPI worker (worker.py)
Agentic:        agent.py — ReAct loop with tools over the same indexes


SCORE PROGRESSION (out of 9)
-------------------------------------------------------------------
VECTOR RAG PATH
v0  Naive baseline (fixed-size chunks, dense-only, no rerank)     4/9
v1  + RecursiveCharacterTextSplitter chunking                     4/9
v2  + Context reordering (compaction)                             5/9
v3  + Metadata filtering (per-company for comparatives)           7/9
v4  + Reranking (candidate_k=35 → final_k=5)                      8/9
    Later side-by-side vs vectorless often counted Vector as 7/9
    depending on Q1 (eval ambiguity). See Q1 notes.

VECTORLESS RAG PATH
v5  Structural nodes + LLM TOC nav + intro node + 2-pass retry    9/9

ADAPTIVE RAG
v6  classify_query → simple_factual | comparative | structural |
    negative → vector / decomposition / vectorless / safety-net
    (+ reformulate retry on low-confidence factual/comparative)

AGENTIC RAG
v7  ReAct agent with tools: vector_search, vectorless_search,
    get_companies, finish — multi-step tool use (not a fixed route)


WHAT WORKED — KEPT IN FINAL PIPELINES
-------------------------------------
1. RecursiveCharacterTextSplitter chunking
2. Context reordering (compaction) before generation
3. Metadata filtering + query decomposition for multi-company questions
4. Cross-encoder reranking with widened candidate pool (k=35 → 5)
5. Vectorless structural TOC navigation + two-pass retry
6. Adaptive query-type routing (language-only classifier)
7. Disk-persistent Qdrant + corpus_index.json (faster cold starts)
8. Two-stage semantic cache (raw + resolved) with topic gate
9. Episodic memory recall/save in Qdrant
10. Agentic ReAct RAG (LLM chooses tools step-by-step)


WHAT WAS TESTED AND NOT KEPT (VECTOR RESEARCH)
----------------------------------------------
1. Hybrid BM25 + RRF — invalid then honest tests on Q3; not kept
2. HyDE — hypothetical named entity correct, retrieval still weak; not kept
3. Dedup — zero near-duplicates at chunk_size=800/overlap=100; not needed
4. RAGAS — dependency conflict; custom IR metrics used instead

(Full write-ups of these experiments remain in the narrative below /
 earlier README history and in NaiveRag_Optimized.ipynb.)


AGENTIC RAG (CURRENT)
---------------------
File: agent.py
Endpoint: POST /agentic (worker.py)
UI: Streamlit pipeline option "Agentic RAG"

Pattern: ReAct (Reason + Act), max ~6 steps.

Tools:
  vector_search(query, company?)     — dense + rerank / filtered search
  vectorless_search(query, company?) — LLM TOC navigation path
  get_companies()                    — list corpus companies
  finish(answer)                     — stop and return final answer

Layers around the loop (same ideas as Adaptive path):
  1. Raw cache check
  2. Resolve query (chat history) + resolved cache
  3. Episodic recall
  4. react_loop(...)
  5. Save cache + episode only if answer is confident
     (does not cache "I don't have enough information")

UI shows for agentic:
  cache_stage, episodes_used, cycles, tool_calls / thoughts
  (transparency into multi-step behavior)


ADAPTIVE vs AGENTIC
-------------------
Adaptive:  ONE classifier decision → fixed retrieval recipe
Agentic:   MULTIPLE LLM decisions → may call vector and/or vectorless
           repeatedly, optionally per company, then finish

Adaptive is cheaper/faster for known query shapes.
Agentic is more flexible when the question needs multi-step gathering.


ARCHITECTURE (DEMO)
-------------------
Terminal 1 — worker (models stay warm):
  uvicorn / python -m uvicorn worker:app --host 127.0.0.1 --port 8000
  init_pipelines():
    load embedder + reranker
    open Qdrant path=qdrant_data (build index only if empty)
    load/build corpus_index.json
    setup episodic collection + load cache_store.json

Terminal 2 — UI:
  streamlit run app.py
  Adaptive / Vector / Vectorless → POST /ask
  Agentic → POST /agentic
  Compare mode → POST /compare (selectable pipelines)

Requires: .env with OPENAI_API_KEY=...


REPO / LOCAL FILES
------------------
pipelines.py   — vector, vectorless, adaptive, cache, episodic, init
agent.py       — ReAct agentic RAG
worker.py      — FastAPI: /health /cache_stats /ask /agentic /compare
app.py         — Streamlit UI
eval_set.json  — 9 eval questions
filtered_2020_filings.parquet
corpus_index.json, qdrant_data/, cache_store.json  (local runtime; gitignore)
NaiveRag_Optimized.ipynb — full research notebook (LangGraph / cache / memory
                          experiments; not all wired into the demo)


HOW TO RUN
----------
cd c:\RAGproject

# Terminal 1
python -m uvicorn worker:app --host 127.0.0.1 --port 8000

# Terminal 2
streamlit run app.py

First worker start loads models (always).
First-ever index build creates qdrant_data + corpus_index (slow once).
Later worker starts reuse disk indexes; models still reload into RAM.


EVAL RESULTS (SUMMARY)
----------------------
| Pipeline        | Score / note                          |
|-----------------|----------------------------------------|
| Vector RAG      | 7/9 or 8/9 (Q1 scoring ambiguity)      |
| Vectorless RAG  | 9/9                                    |
| Adaptive RAG    | routes by query type                   |
| Agentic RAG     | multi-step tool use (demo transparency)|


EVAL-SET QUALITY ISSUE (Q1)
---------------------------
Q1 Microsoft "key competitive risk" is under-specified; multiple valid
grounded answers exist in the filing. Treated as eval design ambiguity,
not a pure pipeline failure.


KEY METHODOLOGICAL LESSON
-------------------------
Measure every change on a fixed eval set. Techniques that sounded right
(hybrid, HyDE, chunking alone) were excluded with evidence; techniques
that moved scores (reorder, metadata filter, rerank, vectorless) were kept.

Ops lesson: keep heavy local models in a long-lived worker; persist vector
indexes; use cache for repeat questions; use episodic memory for related
follow-ups; use Adaptive for cheap routing and Agentic when multi-step
tool choice is the point of the demo.


OPTIONAL NEXT STEPS
-------------------
- Add Agentic to compare multiselect in Streamlit
- Store query_type inside cache entries so route shows on cache hits
- Semantic fact memory (notebook) into worker if desired
- Eval scorecard for Agentic on the same 9 questions