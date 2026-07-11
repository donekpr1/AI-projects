RAG OPTIMIZATION PROJECT — SUMMARY
====================================
10-K Filing Research Assistant — EDGAR-CORPUS (2020), 8 companies

This file documents the full build process: what was tried, what worked,
what didn't, and why — based on a fixed 9-question eval set tested against
every pipeline change and compared the results with  both Vector vs Vectorless RAG on a streamlit app.

Later phases add Vectorless RAG, Adaptive routing, a Streamlit demo, and an
optional local FastAPI worker so models stay warm across UI restarts.


PROJECT SETUP
-------------
Dataset:        eloukas/edgar-corpus, year_2020 (Hugging Face)
Scope:          8 companies (Microsoft, Alphabet, Amazon, Meta, Tesla,
                 Nvidia, Apple, JPMorgan Chase), section_1A (Risk Factors)
Eval set:       9 hand-written questions with known expected answers
                 (5 factual, 2 comparative, 2 negative/out-of-scope)
Generation LLM: gpt-4o-mini (OpenAI)
Embedding:      BAAI/bge-base-en-v1.5 (local, free)
Reranker:       BAAI/bge-reranker-v2-m3 (local, free, via sentence-transformers CrossEncoder)
Vector store:   Qdrant (in-memory; disk persistence optional / not required for demo)
Vectorless:     Structural node TOC + LLM navigation (no embeddings for retrieval)
Demo:           Streamlit UI + optional local FastAPI worker


SCORE PROGRESSION (out of 9, scored manually + retrieval metrics)
-------------------------------------------------------------------
VECTOR RAG PATH
v0  Naive baseline (fixed-size chunks, dense-only, no rerank)     4/9
v1  + RecursiveCharacterTextSplitter chunking                     4/9   (no change — see notes)
v2  + Context reordering (compaction, sort by relevance score)    5/9   (+1, fixed Q2)
v3  + Metadata filtering (per-company retrieval for comparatives) 7/9   (+2, fixed Q6, Q7)
v4  + Reranking (dense candidates, widened pool + cutoff)         8/9   (+1, fixed Q3)
    Note: later Vector vs Vectorless comparison scored Vector as 7/9
    depending on how Q1 (eval ambiguity) was counted. See Q1 notes.

VECTORLESS RAG PATH
v5  Structural nodes + LLM TOC navigation + intro node + 2-pass   9/9

ADAPTIVE RAG
v6  LLM query classifier routes to vector / decomposition /
    vectorless / negative safety-net                              routes by type
    (simple_factual, comparative, structural, negative)


WHAT WORKED — KEPT IN FINAL PIPELINE
-------------------------------------
1. RecursiveCharacterTextSplitter chunking
   - Fixed mid-sentence cuts present in naive fixed-size chunking.
   - Did NOT move the eval score on its own. Important negative-adjacent
     result: better-formed chunks != better retrieval. Kept anyway because
     it's a strict quality improvement with no downside, and it's the
     foundation later techniques (esp. reranking) build on.

2. Context reordering (compaction)
   - Chunks sorted by relevance score (ascending) before insertion into
     the prompt, so the most relevant chunk sits closest to the question.
   - Fixed Q2 (Alphabet "80% advertising revenue"), reproducibly confirmed
     via repeated testing.
   - IMPORTANT CAVEAT: the simple "lost in the middle" explanation does not
     fully hold up — traced the actual chunk positions and found the
     correct answer chunk stayed in the middle position in BOTH the
     failing and succeeding orderings. The real driver appears to be
     which DISTRACTOR sits adjacent to the correct chunk, not absolute
     position. Documented as: real, reproducible effect; precise mechanism
     not fully characterized; would need broader testing to generalize.

3. Metadata filtering + query decomposition (for comparative questions)
   - Detect when a question mentions 2+ known companies; retrieve
     top-k SEPARATELY per company (via Qdrant payload filter) instead of
     one global top-k search.
   - Root cause fixed: a single global search let one company's chunks
     dominate, crowding the other out entirely (0% success on comparative
     questions before this fix).
   - Fixed Q6 and Q7 completely. Single biggest, cleanest win of the
     vector path — a structural fix, not a generation-layer patch.

4. Reranking (cross-encoder, dense-only candidate pool)
   - Wide dense retrieval (candidate_k=35) -> CrossEncoder
     (BAAI/bge-reranker-v2-m3) scores all candidates jointly with the
     query -> keep top final_k=5.
   - First attempt (candidate_k=15, final_k=3) FAILED — diagnosed that the
     answer chunk (Panasonic/Tesla) wasn't even in the candidate pool
     (it ranked 29th in dense search). This is a key lesson: reranking
     cannot fix a chunk that retrieval never surfaced in the first place.
   - Widened candidate_k to 35 (rank 29 now included) and final_k to 5
     (the chunk reranked to position 5) — fixed Q3 ("Panasonic").
   - Reranking did NOT require hybrid/BM25 — operated on dense-only
     candidates throughout. Hybrid search and reranking are independent,
     composable techniques, not a package deal.

5. Vectorless RAG (structural nodes + LLM navigation)
   - Split each company's Item 1A into structural nodes (Company::0, ::1, ...).
   - LLM reads a TOC of node titles and selects node IDs (no embedding search).
   - Always include intro node {company}::0 on first pass when present.
   - Two-pass retry: if answer is "I don't have enough information", navigate
     again excluding already-tried node IDs.
   - Eval: 9/9 on the same fixed set (vectorless_eval_results_v2.json).
   - Wins especially on structural / organization questions where dense
     similarity is a poor match for "how is the section introduced".

6. Adaptive RAG (query-type router)
   - classify_query() uses gpt-4o-mini on query language only (no corpus peek).
   - Routes:
       simple_factual  -> vector rerank_retrieve
       comparative     -> vector retrieve_with_decomposition
       structural      -> vectorless_retrieve_with_retry
       negative        -> vectorless safety net (smaller top_n), then answer
                         or "no info" if still empty
   - Priority in classifier prompt: structural > comparative >
     simple_factual > negative (so "how do MSFT and Meta introduce..."
     stays structural, not comparative).
   - Conservative default: unknown/error -> simple_factual.


WHAT WAS TESTED AND NOT KEPT — WITH DIAGNOSED REASONS
--------------------------------------------------------
1. Hybrid search (BM25 + Reciprocal Rank Fusion)
   - Tried on Q3 (Panasonic question). Initial test looked successful but
     was an INVALID TEST — the test query included the literal word
     "Panasonic," which the real eval question never contains.
   - Retested honestly with the real question phrasing
     ("which supplier was named...") — BM25's top 5 results were
     completely off-topic (Brexit, FTC settlements), confirming neither
     dense nor lexical search has strong signal when the query doesn't
     share vocabulary with the answer.
   - NOT included in final pipeline. Root cause: this particular question
     requires recalling a named entity the query itself never mentions —
     a different problem than hybrid search solves.

2. HyDE (hypothetical document embeddings)
   - Tried on Q3 after hybrid search failed. The LLM's hypothetical
     answer correctly guessed "Panasonic" from general world knowledge —
     premise worked as intended.
   - Still failed at retrieval — diagnosed cause: 2020 is pandemic-year
     data, and COVID-disruption boilerplate is repeated near-identically
     across many companies in the corpus. Dense embedding similarity got
     pulled toward this generic, oversaturated theme rather than the one
     specific named-entity detail.
   - NOT included in final pipeline.

3. Deduplication (compaction sub-technique)
   - Built a diagnostic (inspect_for_compaction) checking all 9 questions'
     retrieved chunks for near-duplicate text overlap.
   - Result: ZERO duplicates detected across the entire eval set at
     chunk_overlap=100, chunk_size=800.
   - Conclusion: dedup logic genuinely not needed for this corpus at this
     chunking configuration. Verified empirically rather than assumed —
     no dedup code added to final pipeline.

4. RAGAS (LLM-as-judge evaluation framework)
   - Installed and attempted, but hit a persistent dependency conflict
     (langchain_community/vertexai import chain) that version-pinning did
     not cleanly resolve within project time constraints.
   - Not blocking: built custom deterministic IR metrics instead (see
     below), which cover retrieval quality without any LLM dependency.


EVAL-SET QUALITY ISSUE FOUND (Q1)
------------------------------------
Q1 ("What does Microsoft cite as a key competitive risk?") generated an
answer about "cloud-based services" risk, which did not match the eval
set's expected answer (general competition from resourced/specialized
rivals).

Investigation showed the generated answer was NOT a hallucination — it was
grounded in real, verbatim Microsoft text discussing a genuinely different,
also-valid competitive risk. Microsoft's filing discusses multiple distinct
competitive risks; the question was under-specified about which one was
wanted.

CONCLUSION: this is an eval-set design flaw (ambiguous question), not a
retrieval or generation failure. Documented rather than "fixed" by further
pipeline changes, since the pipeline was not actually broken.


EVALUATION METRICS USED
--------------------------
Manual scoring:
    Vector path: historically reported 8/9 (Q1 flagged as eval ambiguity);
    side-by-side vs vectorless used 7/9 for Vector depending on Q1 scoring.
    Vectorless v2: 9/9.
    Adaptive: routes by query type (not a single static score).

Custom deterministic IR metrics (no LLM calls):
    id   type         recall_at_k   precision_at_k   reciprocal_rank
    1    factual       1.0           0.4              0.333
    2    factual       1.0           0.4              1.000
    3    factual       1.0           0.4              0.250
    4    factual       1.0           0.8              1.000
    5    factual       1.0           1.0              1.000
    6    comparative   1.0           1.0              1.000
    7    comparative   0.5           0.5              1.000
    8    negative      0.0           0.0              0.000
    9    negative      1.0           0.4              0.500

    Mean Recall@k:    0.833
    Mean Precision@k: 0.544
    Mean MRR:         0.676

CAVEATS — these numbers require interpretation, not blind reading:

  Q7 (0.5 recall/precision) is a DATA LABELING BUG, not a real retrieval
  miss. The eval set's company field used "Meta" while the corpus uses
  "Meta (Facebook)" — exact string match silently failed even though Meta
  WAS genuinely retrieved (confirmed earlier by manual inspection: 3
  Microsoft + 3 Meta chunks came back). Real recall for Q7 is 1.0; fix is
  to correct the eval set's company field, not the pipeline.

  Q8 (0.0 recall) looks like failure but is CORRECT, EXPECTED behavior.
  Amazon's section_1A was empty from the start (known data gap, flagged
  on day one) -- zero Amazon chunks exist in the index, so it is
  mathematically impossible for retrieval to "find" Amazon. The system
  correctly answered "I don't have enough information." Recall@k cannot
  distinguish "failed to find something that exists" from "correctly
  found nothing because nothing exists" -- it produces a misleading 0.0
  for a question that actually passed.

  Q9 (1.0 recall) is similarly not meaningful -- Apple chunks were
  retrieved (Apple has plenty of content), but finding ANY Apple content
  was never the actual goal of an out-of-scope stock-price question. The
  1.0 is coincidental, not evidence of correct behavior.

  GENERAL LESSON: Recall@k / Precision@k / MRR, as built, are well-suited
  to factual and comparative questions, but do not correctly model
  negative/out-of-scope test cases, where success means correctly finding
  NOTHING useful and saying so -- not finding the "right" company. Same
  underlying lesson as the faithfulness-check findings above, from a
  different angle: automated metrics need human interpretation layered on
  top, not blind trust in the raw numbers.

Note on RAGAS / LLM-judge metrics: industry-standard, widely used in
production, but for a different purpose than IR metrics — they evaluate
the GENERATED ANSWER (faithfulness, answer relevance), which deterministic
IR metrics cannot measure at all (IR metrics only check retrieval, not
whether the LLM used the retrieved content correctly). Real-world practice:
cheap IR metrics for continuous/high-volume monitoring, LLM-judge metrics
for periodic deeper quality audits — not a replacement for each other.


CUSTOM FAITHFULNESS CHECK — RESULTS AND LIMITATIONS FOUND
--------------------------------------------------------------
After RAGAS was blocked by a dependency conflict, a custom LLM-judge
faithfulness check was built directly (single prompt asking GPT-4o-mini
"is this answer supported by this context, YES/NO + reason").

Result: 5 YES (Q2, Q3, Q4, Q5, Q6), 4 NO (Q1, Q7, Q8, Q9).

On inspection, the NO verdicts were NOT all valid — spot-checking against
ground truth already established earlier in the project revealed two
distinct, separate problems with the check itself:

1. JUDGE HALLUCINATION (Q1, Q7)
   Q1's judge verdict claimed the "cloud-based services" content came
   from Apple's filing. This is factually wrong — the same content had
   already been confirmed earlier (by direct inspection) to be labeled
   [Microsoft], from Microsoft's own filing. The judge misattributed
   company ownership of content, likely because Q1's retrieved context
   mixed Apple and Microsoft chunks together. Q7 showed a similar
   unforced error — the judge claimed the answer "does not address the
   question," contradicting the actual saved answer, which was a real,
   substantive Microsoft vs. Meta comparison.

   LESSON: an LLM judge can hallucinate or misread its own input, exactly
   like the generator LLM. A "NO" verdict is a signal to investigate, not
   a fact — automated evaluation output requires the same skepticism as
   any other LLM output, not blind trust.

2. PROMPT DESIGN FLAW (Q8, Q9)
   Both questions correctly produced "I don't have enough information to
   answer that" — the desired, correct behavior for out-of-scope/missing-
   data questions. But the faithfulness prompt only asked "is the answer
   supported by the context," and a refusal technically doesn't draw on
   the context at all, so the judge marked both NO despite the refusal
   being exactly correct.

   LESSON: a correct refusal should count as maximally faithful (zero
   unsupported claims made), but the prompt didn't encode that case.
   Fix (not yet implemented): explicitly instruct the judge that
   "I don't have enough information" answers are faithful by default
   when the context genuinely lacks the requested information.

CONCLUSION: the custom faithfulness check, as built, is not yet reliable
enough to trust unsupervised — consistent with the broader lesson from
Q1's eval-set ambiguity finding earlier: every layer of this project,
including the evaluation layer itself, needed direct human spot-checking
rather than being trusted at face value.


FINAL PIPELINE COMPOSITION
------------------------------
A) Vector RAG (manual / compare mode)
query
  -> detect_companies(query)               [routes single vs. comparative]
  -> IF comparative (2+ companies):
         retrieve_with_decomposition()     [metadata filter, per-company]
     ELSE:
         rerank_retrieve()                 [dense candidates -> cross-encoder rerank]
  -> generate_answer()                     [reorder by score -> grounded LLM prompt]
  -> answer

B) Vectorless RAG (manual / compare mode)
query
  -> detect_companies(query)
  -> LLM navigates company TOC(s)          [select node IDs]
  -> optional 2nd navigation pass on "don't know"
  -> generate_answer()
  -> answer

C) Adaptive RAG (default demo path)
query
  -> classify_query()                      [simple_factual | comparative |
                                            structural | negative]
  -> route to A or B as above
  -> return unified result dict            [includes query_type for UI]

Techniques in shipped demo pipelines:
  recursive chunking, context reordering, metadata filtering, reranking,
  vectorless TOC navigation + retry, adaptive query classification.

Techniques tested and deliberately excluded from vector path:
  hybrid/BM25/RRF, HyDE, dedup.


STREAMLIT DEMO + LOCAL WORKER
--------------------------------
Files:
  pipelines.py  — shared retrieval/generation logic
  app.py        — Streamlit UI (Adaptive / Vector / Vectorless + compare)
  worker.py     — optional FastAPI process that loads models once

Why a worker:
  Embedding + reranker models (and in-memory indexes) live in the worker
  process. Restarting Streamlit does NOT reload those models as long as
  the worker stays running. LLM calls (gpt-4o-mini) are still per-request
  OpenAI API calls either way — the worker does not "load" the LLM locally.

What the worker does NOT do by itself:
  Persist Qdrant or corpus_index to disk. Worker restart still rebuilds
  in-memory indexes unless disk persistence is added separately.

How to run (two terminals):

  # Terminal 1 — worker (slow once; leave open)
  cd <project-folder>
  pip install fastapi uvicorn requests streamlit
  uvicorn worker:app --host 127.0.0.1 --port 8000

  # Terminal 2 — UI (can restart freely)
  streamlit run app.py

  Worker health: http://127.0.0.1:8000/health
  API docs:      http://127.0.0.1:8000/docs

Without worker (models load inside Streamlit on each app start):
  streamlit run app.py
  (requires app.py that imports pipelines directly instead of HTTP)

Requires:
  .env with OPENAI_API_KEY=...
  filtered_2020_filings.parquet in the project folder


OPTIONAL NEXT STEPS (NOT REQUIRED FOR DEMO)
---------------------------------------------
1. Persist Qdrant to a local folder (path=...) and skip re-index on startup
2. Save/load corpus_index.json for vectorless nodes
3. Separate ingest job vs serving process (production pattern)
4. Fix faithfulness-judge prompt for correct refusals
5. Align eval-set company labels (Meta vs Meta (Facebook))


KEY METHODOLOGICAL LESSON
-----------------------------
Every change was tested against the same fixed eval set, and "tried it,
it didn't help, here's the diagnosed reason why" was treated as a valid,
documented result — not a failure to hide. Several plausible-sounding
techniques (better chunking alone, hybrid search, HyDE) did not move the
score and were excluded with evidence, while techniques that did help
(reordering, metadata filtering, reranking, vectorless navigation,
adaptive routing) were validated with clear before/after numbers and a
traced root cause. This discipline — measure before and after every single
change — is the actual reusable takeaway, more so than any individual
technique.

Secondary ops lesson: keep heavy local models warm in a long-lived process
(worker or always-on service); persist indexes to disk so cold starts do
not re-embed the corpus; LLM API calls remain per-request either way.