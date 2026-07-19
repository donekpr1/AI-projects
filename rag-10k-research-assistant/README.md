# RAG OPTIMIZATION PROJECT — SUMMARY

10-K Filing Research Assistant — EDGAR-CORPUS (2020), 8 companies

This file documents the full build process: what was tried, what worked,
what didn't, and why — based on a fixed 9-question eval set tested against
every pipeline change.

Shipped demo stack (current):
Auto (Adaptive + Vectorless fallback) · Adaptive · Vector · Vectorless · Agentic (ReAct)
Streamlit UI · FastAPI worker · disk Qdrant · semantic cache · episodic memory



## PROJECT SETUP

Dataset:        eloukas/edgar-corpus, year\_2020 (Hugging Face)
Scope:          8 companies (Microsoft, Alphabet, Amazon, Meta, Tesla,
Nvidia, Apple, JPMorgan Chase), section\_1A (Risk Factors)
Note: Amazon section\_1A is empty in this slice (known gap).
Eval set:       9 hand-written questions with known expected answers
(5 factual, 2 comparative, 2 negative/out-of-scope)
Generation LLM: gpt-4o-mini (OpenAI)
Embedding:      BAAI/bge-base-en-v1.5 (local)
Reranker:       BAAI/bge-reranker-v2-m3 (local, CrossEncoder)
Vector store:   Qdrant on disk (qdrant\_data/) — rebuilt only if missing
Vectorless:     corpus\_index.json (structural nodes) + LLM TOC navigation
Cache:          raw + resolved semantic cache (cache\_store.json) — shared across pipelines
Memory:         episodic Q\&A in Qdrant collection episodic\_memory
Demo:           Streamlit (app.py) + FastAPI worker (worker.py)
Agentic:        agent.py — ReAct loop with tools over the same indexes
Eval script:    eval\_agentic.py — cold gold-set scoring for Agentic (+ optional Adaptive)



## SCORE PROGRESSION (out of 9)

VECTOR RAG PATH
v0  Naive baseline (fixed-size chunks, dense-only, no rerank)     4/9
v1  + RecursiveCharacterTextSplitter chunking                     4/9
v2  + Context reordering (compaction)                             5/9
v3  + Metadata filtering (per-company for comparatives)           7/9
v4  + Reranking (candidate\_k=35 → final\_k=5)                      8/9
Later side-by-side vs vectorless often counted Vector as 7/9
depending on Q1 (eval ambiguity). See Q1 notes.

VECTORLESS RAG PATH
v5  Structural nodes + LLM TOC nav + intro node + 2-pass retry    9/9

ADAPTIVE RAG
v6  classify\_query → simple\_factual | comparative | structural |
negative → vector / decomposition / vectorless / safety-net
(+ reformulate retry on low-confidence factual/comparative)

AGENTIC RAG
v7  ReAct agent with tools: vector\_search, vectorless\_search,
get\_companies, finish — multi-step tool use (not a fixed route)
v7b Fail-fast budget + one Vectorless fallback on don't-know
(keeps Agentic from burning long vector loops on hard entities)

AUTO (DEFAULT UI PATH)
v8  Cache → Adaptive once → if don't-know, Vectorless once
(never defaults to Agentic — cost/latency)



## WHAT WORKED — KEPT IN FINAL PIPELINES

1. RecursiveCharacterTextSplitter chunking
2. Context reordering (compaction) before generation
3. Metadata filtering + query decomposition for multi-company questions
4. Cross-encoder reranking with widened candidate pool (k=35 → 5)
5. Vectorless structural TOC navigation + two-pass retry
6. Adaptive query-type routing (language-only classifier)
7. Disk-persistent Qdrant + corpus\_index.json (faster cold starts)
8. Two-stage semantic cache (raw + resolved) with topic gate
9. Episodic memory recall/save in Qdrant
10. Agentic ReAct RAG (LLM chooses tools step-by-step)
11. Comparative generation: allow\_compare synthesizes across company-tagged
    chunks (no need for one pre-compared document)
12. Skip episodic injection on comparative / multi-company queries
    (episodes still saved; avoids don't-know bias)
13. Auto cheap fallback: Adaptive → Vectorless only on failure



## WHAT WAS TESTED AND NOT KEPT (VECTOR RESEARCH)

1. Hybrid BM25 + RRF — invalid then honest tests on Q3; not kept
2. HyDE — hypothetical named entity correct, retrieval still weak; not kept
3. Dedup — zero near-duplicates at chunk\_size=800/overlap=100; not needed
4. RAGAS — dependency conflict; custom IR metrics used instead

(Full write-ups of these experiments remain in the narrative below /
earlier README history and in NaiveRag\_Optimized.ipynb.)



## AUTO (ADAPTIVE + FALLBACK)

Default Streamlit pipeline. Production-style cheap path:

1. Semantic cache (raw → resolve → resolved)
2. Adaptive RAG once (classifier picks one retrieval recipe)
3. If don't-know / low confidence → Vectorless once
4. Never runs Agentic on this path

UI caption: fallback none | vectorless | vectorless\_failed



## AGENTIC RAG (CURRENT)

File: agent.py
Endpoint: POST /agentic (worker.py)
UI: Streamlit pipeline option "Agentic RAG"

Pattern: ReAct (Reason + Act), max 4 steps (fail-fast).
If the primary pass is don't-know / soft refusal → one Vectorless
fallback (per detected company, max 2), then synthesize and return.

Tools:
vector\_search(query, company?)     — dense + light rerank / company filter
vectorless\_search(query, company?) — LLM TOC navigation path
get\_companies()                    — list corpus companies
finish(...)                        — stop; system synthesizes from observations

Layers around the loop (same ideas as Adaptive path):

1. Raw cache check
2. Resolve query (chat history) + resolved cache
3. Episodic recall (skipped for comparative / multi-company injection)
4. react\_loop(...)
5. Optional Vectorless fallback on weak answer
6. Save cache + episode only if confident and not a soft refusal
   (avoids poisoning the shared cache)

UI shows for agentic:
cache\_stage, episodes\_used, cycles, tool\_calls / thoughts, fallback
(transparency into multi-step behavior)

Known limit: buried named entities (e.g. Q3 Panasonic) remain hard for
dense Agentic search; Vectorless / Auto are stronger defaults for quality.



## ADAPTIVE vs AGENTIC vs AUTO

Adaptive:  ONE classifier decision → fixed retrieval recipe
Agentic:   MULTIPLE LLM decisions → may call vector and/or vectorless,
           then optional Vectorless fallback on failure
Auto:      Adaptive first; Vectorless only if Adaptive fails (no Agentic)

Adaptive / Auto are cheaper/faster for known query shapes.
Agentic is optional when multi-step tool choice is the demo point.



## ARCHITECTURE (DEMO)

Terminal 1 — worker (models stay warm):
python -m uvicorn worker:app --host 127.0.0.1 --port 8000
init\_pipelines():
load embedder + reranker
open Qdrant path=qdrant\_data (build index only if empty)
load/build corpus\_index.json
setup episodic collection + load cache\_store.json

Terminal 2 — UI:
streamlit run app.py
Auto / Adaptive / Vector / Vectorless → POST /ask
Agentic → POST /agentic
Compare mode → POST /compare (selectable pipelines)

Requires: .env with OPENAI\_API\_KEY=...

Important: only ONE worker may open qdrant\_data at a time.
A second uvicorn gets a lock / permission error — stop the first process first.



## REPO / LOCAL FILES

pipelines.py      — vector, vectorless, adaptive, auto fallback, cache, episodic, init
agent.py          — ReAct agentic RAG (+ Vectorless fail-fast fallback)
worker.py         — FastAPI: /health /cache\_stats /ask /agentic /compare
app.py            — Streamlit UI
eval\_set.json     — 9 eval questions (expected\_answer gold)
eval\_agentic.py   — run Agentic (and optional Adaptive) against gold set
filtered\_2020\_filings.parquet
corpus\_index.json, qdrant\_data/, cache\_store.json  (local runtime; gitignore)
agentic\_eval\_results.json — written by eval\_agentic.py
NaiveRag\_Optimized.ipynb — full research notebook (LangGraph / cache / memory
experiments; not all wired into the demo)



## HOW TO RUN

cd c:\\RAGproject

# Terminal 1 (only one worker)

python -m uvicorn worker:app --host 127.0.0.1 --port 8000

# Terminal 2

streamlit run app.py

First worker start loads models (always).
First-ever index build creates qdrant\_data + corpus\_index (slow once).
Later worker starts reuse disk indexes; models still reload into RAM.

# Optional — Agentic gold eval (worker must be up; clear cache for cold run)

del cache\_store.json
python eval\_agentic.py
python eval\_agentic.py --compare-adaptive
python eval\_agentic.py --ids 3,6,8



## EVAL RESULTS (SUMMARY)

| Pipeline | Score / note |
|----------|----------------|
| Vector RAG | 7/9 or 8/9 (Q1 scoring ambiguity) |
| Vectorless RAG | 9/9 |
| Adaptive RAG | routes by query type |
| Auto | Adaptive → Vectorless on don't-know |
| Agentic RAG | multi-step tools; optional Vectorless fallback; weaker/slower on buried entities (e.g. Q3) — re-score with eval\_agentic.py |

IR metrics (Recall@k / Precision@k / MRR) were computed in the notebook
(company-level), not in the live worker. Answer correctness vs gold is
manual / eval\_agentic LLM-judge, not an automatic gate on every /ask.



## EVAL-SET QUALITY ISSUE (Q1)

Q1 Microsoft "key competitive risk" is under-specified; multiple valid
grounded answers exist in the filing. Treated as eval design ambiguity,
not a pure pipeline failure.



## KEY METHODOLOGICAL LESSON

Measure every change on a fixed eval set. Techniques that sounded right
(hybrid, HyDE, chunking alone) were excluded with evidence; techniques
that moved scores (reorder, metadata filter, rerank, vectorless) were kept.

Ops lesson: keep heavy local models in a long-lived worker; persist vector
indexes; use cache for repeat questions; use episodic memory for related
follow-ups; use Auto/Adaptive for cheap routing and Agentic when multi-step
tool choice is the point of the demo. One local Qdrant path = one worker.
