"""RAG pipelines for Streamlit — extracted from NaiveRag_Optimized.ipynb."""
import re
import json
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance, PointStruct,
    Filter, FieldCondition, MatchValue
)
from sentence_transformers import SentenceTransformer, CrossEncoder

ROOT = Path(__file__).parent
COLLECTION = "risk_factors_recursive"
DONT_KNOW_PHRASE = "i don't have enough information"

load_dotenv()
llm = OpenAI()

filtered = pd.read_parquet(ROOT / "filtered_2020_filings.parquet")
COMPANY_NAMES = filtered["company_name"].unique().tolist()

embedder = None
client = None
reranker = None
corpus_index = None

# ── Valid query types for adaptive RAG ───────────────────────
VALID_QUERY_TYPES = ["simple_factual", "comparative", "structural", "negative"]


# ── ALL YOUR EXISTING FUNCTIONS (unchanged) ──────────────────

def detect_companies(query, company_names):
    return [
        name for name in company_names
        if name.split()[0].lower() in query.lower()
    ]


def generate_answer(query, retrieved_chunks):
    """
    Builds prompt from retrieved chunks and calls GPT-4o-mini.
    Works for both Qdrant ScoredPoint objects and VectorlessResult
    objects because both expose .score and .payload["text"] and
    .payload["company"] — same interface, different source.
    Reorders chunks by score (compaction) before building prompt —
    lowest score first, highest last, so most relevant chunk sits
    closest to the question.
    """
    reordered = sorted(retrieved_chunks, key=lambda r: r.score)
    context = "\n\n---\n\n".join(
        f"[{r.payload['company']}]: {r.payload['text']}"
        for r in reordered
    )
    prompt = f"""Answer the question using ONLY the context below.
If the context doesn't contain enough information to answer,
say "I don't have enough information to answer that" rather than guessing.

Context:
{context}

Question: {query}

Answer:"""
    response = llm.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return response.choices[0].message.content


def build_chunks_recursive(filtered_df, chunk_size=800, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunks = []
    for _, row in filtered_df.iterrows():
        text = row["section_1A"]
        if not text or pd.isna(text) or len(str(text).strip()) == 0:
            continue
        for chunk_text in splitter.split_text(str(text)):
            chunks.append({
                "text": chunk_text,
                "company": row["company_name"],
                "cik": row["cik"],
                "section": "section_1A",
            })
    return chunks


def index_chunks(chunks, collection_name):
    global embedder, client
    texts = [c["text"] for c in chunks]
    embeddings = embedder.encode(texts, show_progress_bar=False)
    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=len(embeddings[0]),
            distance=Distance.COSINE
        ),
    )
    points = [
        PointStruct(id=i, vector=embeddings[i].tolist(), payload=chunks[i])
        for i in range(len(chunks))
    ]
    client.upsert(collection_name=collection_name, points=points)
    return len(points)


def rerank_retrieve(collection_name, query, candidate_k=35, final_k=5):
    """
    Single-company retrieval path.
    Wide dense search (candidate_k=35) followed by cross-encoder
    reranking to top final_k=5. candidate_k=35 because Panasonic
    chunk ranked 29th in dense search — needed wide pool for reranker
    to surface it.
    """
    global embedder, client, reranker
    query_vector = embedder.encode(query).tolist()
    candidates = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=candidate_k
    ).points
    pairs = [[query, c.payload["text"]] for c in candidates]
    rerank_scores = reranker.predict(pairs)
    scored = sorted(
        zip(candidates, rerank_scores),
        key=lambda x: x[1],
        reverse=True
    )

    class RerankedResult:
        def __init__(self, payload, score):
            self.payload = payload
            self.score = score

    return [RerankedResult(c.payload, score) for c, score in scored[:final_k]]


def retrieve_with_decomposition(collection_name, query, top_k=3):
    """
    Comparative retrieval path.
    Runs separate metadata-filtered Qdrant search per detected company.
    Returns top_k chunks per company — guarantees equal representation
    regardless of embedding similarity scores.
    Root cause this fixes: single global search let one company sweep
    all top-k slots, giving the other zero representation.
    """
    global embedder, client
    companies = detect_companies(query, COMPANY_NAMES)
    query_vector = embedder.encode(query).tolist()
    if len(companies) >= 2:
        all_results = []
        for company in companies:
            results = client.query_points(
                collection_name=collection_name,
                query=query_vector,
                query_filter=Filter(
                    must=[FieldCondition(
                        key="company",
                        match=MatchValue(value=company)
                    )]
                ),
                limit=top_k,
            )
            all_results.extend(results.points)
        return all_results
    return client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=top_k
    ).points


def final_retrieve(collection_name, query, candidate_k=35, final_k=5):
    """
    Original vector retrieval router — kept for run_vector() path.
    Routes to comparative or single based on company count.
    """
    if len(detect_companies(query, COMPANY_NAMES)) >= 2:
        return retrieve_with_decomposition(collection_name, query, top_k=3)
    return rerank_retrieve(collection_name, query, candidate_k, final_k)


def split_risk_factors_into_nodes(
    section_text, company, min_chars=80, max_chars=1500
):
    if not section_text or not str(section_text).strip():
        return [{
            "id": f"{company}::0",
            "company": company,
            "title": "[EMPTY SECTION]",
            "text": "",
            "char_len": 0,
        }]
    text = str(section_text).strip()
    raw_parts = re.split(
        r'\n(?=\s*(?:Item 1A\.?|ITEM 1A\.?|RISK FACTORS|•|\(\d+\)|\d+\.\s+[A-Z]))',
        text,
        flags=re.IGNORECASE,
    )
    nodes = []
    buf = ""

    def flush_buffer():
        nonlocal buf
        chunk = buf.strip()
        buf = ""
        if len(chunk) < min_chars:
            return
        nodes.append({
            "id": f"{company}::{len(nodes)}",
            "company": company,
            "title": chunk[:120].replace("\n", " ").strip(),
            "text": chunk,
            "char_len": len(chunk),
        })

    for part in raw_parts:
        part = part.strip()
        if not part:
            continue
        if len(part) > max_chars:
            if buf:
                flush_buffer()
            for i in range(0, len(part), max_chars - 100):
                sub = part[i:i + max_chars]
                if len(sub) >= min_chars:
                    nodes.append({
                        "id": f"{company}::{len(nodes)}",
                        "company": company,
                        "title": sub[:120].replace("\n", " ").strip(),
                        "text": sub,
                        "char_len": len(sub),
                    })
        else:
            if len(buf) + len(part) + 1 > max_chars:
                flush_buffer()
            buf = f"{buf}\n{part}".strip() if buf else part
    if buf:
        flush_buffer()
    if not nodes:
        nodes.append({
            "id": f"{company}::0",
            "company": company,
            "title": text[:120].replace("\n", " "),
            "text": text,
            "char_len": len(text),
        })
    return nodes


def build_corpus_index(filtered_df):
    return {
        row["company_name"]: split_risk_factors_into_nodes(
            row["section_1A"], row["company_name"]
        )
        for _, row in filtered_df.iterrows()
    }


def format_toc(nodes, exclude_ids=None, max_title_len=100):
    exclude_ids = set(exclude_ids or [])
    lines = []
    for n in nodes:
        if n["id"] in exclude_ids:
            continue
        if n["char_len"] == 0:
            lines.append(f"- {n['id']}: [EMPTY SECTION]")
        else:
            lines.append(
                f"- {n['id']}: {n['title'][:max_title_len]} ({n['char_len']} chars)"
            )
    return "\n".join(lines)


def parse_node_id_list(raw_text):
    raw_text = raw_text.strip()
    if "```" in raw_text:
        raw_text = re.sub(r"```(?:json)?", "", raw_text).strip("` \n")
    start = raw_text.find("[")
    end = raw_text.rfind("]")
    if start == -1 or end == -1:
        return []
    try:
        ids = json.loads(raw_text[start:end + 1])
        return [str(i) for i in ids if isinstance(i, (str, int))]
    except json.JSONDecodeError:
        return []


class VectorlessResult:
    def __init__(self, payload, score=1.0, node_id=None):
        self.payload = payload
        self.score = score
        self.node_id = node_id


def ensure_intro_node(selected_ids, company, nodes):
    intro_id = f"{company}::0"
    node_map = {n["id"]: n for n in nodes}
    if (
        intro_id in node_map
        and node_map[intro_id]["char_len"] > 0
        and intro_id not in selected_ids
    ):
        selected_ids = [intro_id] + selected_ids
    return selected_ids


def navigate_company_tree(
    query, company, nodes, top_n=5, exclude_ids=None, pass_label="first"
):
    if not nodes:
        return []
    if len(nodes) == 1 and nodes[0]["char_len"] == 0:
        return [VectorlessResult(
            payload={
                "company": company,
                "text": "[This company's risk factors section is empty in the dataset.]",
            },
            score=0.0,
            node_id=nodes[0]["id"],
        )]

    toc = format_toc(nodes, exclude_ids=exclude_ids)
    exclude_note = ""
    if exclude_ids:
        exclude_note = (
            f"\nDo NOT pick these already-tried node IDs: {list(exclude_ids)}\n"
            "Pick different nodes that might contain the answer.\n"
        )

    nav_prompt = f"""You are an expert reading SEC 10-K Item 1A (Risk Factors) for {company}.

This is the {pass_label} navigation pass.
Given the question, select up to {top_n} node IDs most likely to contain the answer.

Rules:
- Return ONLY a JSON list of node ID strings, e.g. ["{company}::0", "{company}::2"]
- Pick the most specific nodes for the question
- For intro/cross-reference/structural questions, include early nodes like {company}::0
- For supplier/entity questions, pick nodes about supply chain, partners, or named companies
- Do not invent IDs — only use IDs from the list below
{exclude_note}
Table of contents:
{toc}

Question: {query}
"""
    response = llm.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": nav_prompt}],
        temperature=0,
    )
    selected_ids = parse_node_id_list(response.choices[0].message.content)
    if pass_label == "first" and not exclude_ids:
        selected_ids = ensure_intro_node(selected_ids, company, nodes)

    node_map = {n["id"]: n for n in nodes}
    results = []
    seen = set()
    for rank, nid in enumerate(selected_ids):
        if nid in seen:
            continue
        node = node_map.get(nid)
        if node and node["char_len"] > 0:
            seen.add(nid)
            results.append(VectorlessResult(
                payload={"company": company, "text": node["text"]},
                score=float(rank),
                node_id=nid,
            ))
        if len(results) >= top_n:
            break
    if not results and nodes[0]["char_len"] > 0:
        results.append(VectorlessResult(
            payload={"company": company, "text": nodes[0]["text"]},
            score=0.0,
            node_id=nodes[0]["id"],
        ))
    return results


def vectorless_retrieve(query, corpus_index_map, top_n=5):
    companies = detect_companies(query, COMPANY_NAMES)
    target = companies if companies else list(corpus_index_map.keys())
    all_results = []
    for company in target:
        nodes = corpus_index_map.get(company, [])
        all_results.extend(
            navigate_company_tree(
                query, company, nodes, top_n=top_n, pass_label="first"
            )
        )
    return all_results


def vectorless_retrieve_with_retry(query, corpus_index_map, top_n=5):
    """
    Pass 1: navigate + generate.
    Pass 2: if answer is "don't know", navigate again excluding
            already-tried nodes so LLM picks different sections.
    Second pass results get score offset -10 (lower score = higher
    priority) so they appear first in merged results.
    """
    first_results = vectorless_retrieve(query, corpus_index_map, top_n=top_n)
    first_answer = generate_answer(query, first_results)
    if DONT_KNOW_PHRASE not in first_answer.lower():
        return first_results, first_answer, 1

    companies = detect_companies(query, COMPANY_NAMES)
    target = companies if companies else list(corpus_index_map.keys())
    tried_ids = {r.node_id for r in first_results if r.node_id}
    second_results = []
    for company in target:
        nodes = corpus_index_map.get(company, [])
        second_results.extend(
            navigate_company_tree(
                query, company, nodes,
                top_n=top_n,
                exclude_ids=tried_ids,
                pass_label="second",
            )
        )

    merged = [
        VectorlessResult(r.payload, score=r.score - 10, node_id=r.node_id)
        for r in second_results
    ]
    merged += [
        VectorlessResult(r.payload, score=r.score, node_id=r.node_id)
        for r in first_results
    ]
    best_by_id = {}
    for r in merged:
        if r.node_id not in best_by_id or r.score < best_by_id[r.node_id].score:
            best_by_id[r.node_id] = r
    final_results = sorted(
        best_by_id.values(), key=lambda r: r.score
    )[: top_n * max(len(target), 1)]
    second_answer = generate_answer(query, final_results)
    return final_results, second_answer, 2


# ── NEW: ADAPTIVE RAG FUNCTIONS ───────────────────────────────

def classify_query(query: str) -> str:
    """
    Classifies the user query into one of four types using GPT-4o-mini.

    Classification is based ONLY on the surface language of the query —
    the LLM never sees the corpus data. It reads your category definitions
    and matches the query's intent to the best fit.

    Conservative design:
      "When in doubt use simple_factual" ensures the classifier defaults
      to the full pipeline rather than skipping valid questions.
      Only returns "negative" when completely certain the topic is
      out of scope (stock prices, addresses, real-time data).
      Only returns "structural" when the question clearly asks about
      document organization — not about fact content.

    Priority: structural > comparative > simple_factual > negative
      Explicitly stated in prompt so LLM resolves ambiguous cases
      (e.g. two companies in a structural question) correctly.

    max_tokens=10: one word only — minimal cost and latency.
    Fallback: unexpected LLM output → "simple_factual" (safe default).
    """
    classify_prompt = f"""Classify this SEC 10-K filing question into exactly one category.

Categories:
  structural     — HIGHEST PRIORITY: asks HOW a section is organized,
                   introduced, or structured. Key signals: "how does X
                   introduce", "how is the section structured", "how do
                   they open their risk factors". TWO companies can appear
                   in a structural question — that does NOT make it
                   comparative. If the question asks about DOCUMENT
                   ORGANIZATION, always classify as structural regardless
                   of how many companies are mentioned.
                   Example: "How do Microsoft and Meta introduce their
                   risk factor sections?" → structural

  comparative    — asks to COMPARE FACTS OR CONTENT between companies.
                   Key signal: asking about the SUBSTANCE of what companies
                   say, not about how their document is organized.
                   Example: "How do Tesla and JPMorgan's COVID risks differ?"
                   → comparative

  simple_factual — asks for one specific fact about one company.
                   Example: "What percentage of Alphabet's revenue came
                   from advertising?" → simple_factual

  negative       — ONLY use if VERY confident the topic is completely
                   out of scope for a risk factors filing.
                   Examples: stock prices, physical addresses, interview
                   quotes, real-time market data.
                   When in doubt, use simple_factual — never skip
                   retrieval unless completely certain it's out of scope.

PRIORITY ORDER: structural > comparative > simple_factual > negative
If structural signals are present, always pick structural even if
two companies are mentioned.

Return ONLY the category name, nothing else. No explanation, no punctuation.

Question: {query}"""

    try:
        response = llm.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": classify_prompt}],
            temperature=0,
            max_tokens=10,
        )
        classification = response.choices[0].message.content.strip().lower()
        if classification not in VALID_QUERY_TYPES:
            print(f"Unknown classification '{classification}' → defaulting to simple_factual")
            return "simple_factual"
        return classification
    except Exception as e:
        print(f"Classifier error: {e} → defaulting to simple_factual")
        return "simple_factual"


def run_adaptive(query: str) -> dict:
    """
    Main entry point for adaptive RAG.

    Step 1 — classify_query():
      Sends query to GPT-4o-mini with classification prompt.
      Returns one of: simple_factual, comparative, structural, negative.
      Never checks the corpus — decision based on query language only.

    Step 2 — route based on query_type:
      simple_factual → vector RAG (dense search + reranking)
        Best for: specific factual questions about one company
        Proven on: Q2-Q5 in eval set

      comparative → vector RAG with metadata filtering
        Best for: questions explicitly comparing two companies
        Proven on: Q6 in eval set

      structural → vectorless RAG (LLM navigation over TOC)
        Best for: questions about document organization/structure
        Proven on: Q7 — vectorless wins over vector (True vs False)
        Skips embedding entirely — LLM reads section titles instead

      negative → vectorless safety net first, then "no info"
        Best for: topics out of scope for risk factors filing
        Why vectorless safety net: classifier made decision without
        checking corpus. Vectorless gives data a chance to override
        the classifier's language-only judgment. If vectorless also
        finds nothing → genuinely out of scope → return "no info".

    Step 3 — return unified result dict:
      Same keys regardless of which path ran so app.py can handle
      all four paths with the same display code.
      Includes query_type so UI can show which route was taken.
    """
    t0 = time.perf_counter()

    # Step 1 — classify
    query_type = classify_query(query)

    # Step 2 — route to appropriate retrieval strategy
    if query_type == "simple_factual":
        # Vector RAG — single company path
        # Dense search (candidate_k=35) → cross-encoder reranking → top 5
        retrieved = rerank_retrieve(COLLECTION, query)
        answer = generate_answer(query, retrieved)
        sources = [
            {
                "company": r.payload["company"],
                "label": f"score {float(r.score):.4f}",
                "text": r.payload["text"],
            }
            for r in retrieved
        ]
        navigation_passes = None   # not applicable for vector path

    elif query_type == "comparative":
        # Vector RAG — comparative path
        # Per-company metadata filtered search, top 3 per company
        retrieved = retrieve_with_decomposition(COLLECTION, query, top_k=3)
        answer = generate_answer(query, retrieved)
        sources = [
            {
                "company": r.payload["company"],
                "label": f"score {float(r.score):.4f}",
                "text": r.payload["text"],
            }
            for r in retrieved
        ]
        navigation_passes = None

    elif query_type == "structural":
        # Vectorless RAG — LLM navigation over section TOC
        # No embedding, no Qdrant search for retrieval
        # LLM reads section titles → picks node IDs → fetch full text
        # Two-pass retry if first navigation fails
        retrieved, answer, passes = vectorless_retrieve_with_retry(
            query, corpus_index, top_n=5
        )
        sources = [
            {
                "company": r.payload["company"],
                "label": r.node_id or "section",
                "text": r.payload["text"],
            }
            for r in retrieved
        ]
        navigation_passes = passes

    else:
        # negative — vectorless safety net first
        # Classifier said "negative" but never checked the corpus.
        # Vectorless gives corpus a chance to override the classification.
        # If vectorless finds content → answer it (classifier was wrong)
        # If vectorless also fails → genuinely out of scope → "no info"
        retrieved, answer, passes = vectorless_retrieve_with_retry(
            query, corpus_index, top_n=3  # smaller top_n — low confidence
        )
        sources = [
            {
                "company": r.payload["company"],
                "label": r.node_id or "section",
                "text": r.payload["text"],
            }
            for r in retrieved
        ]
        navigation_passes = passes

    return {
        "pipeline":          "Adaptive RAG",
        "query_type":        query_type,        # which route was taken
        "answer":            answer,
        "elapsed_sec":       round(time.perf_counter() - t0, 2),
        "navigation_passes": navigation_passes,  # None for vector paths
        "companies":         list({s["company"] for s in sources}),
        "sources":           sources,
    }


def init_pipelines():
    """
    Initialises all models and builds all indexes.
    Called once at Streamlit app startup.
    Order matters:
      1. embedder — needed by index_chunks and rerank_retrieve
      2. client — needed by index_chunks and all Qdrant searches
      3. reranker — needed by rerank_retrieve
      4. index_chunks — builds Qdrant vector index (1129 chunks)
      5. corpus_index — builds Python dict for vectorless path
    """
    global embedder, client, reranker, corpus_index
    embedder = SentenceTransformer("BAAI/bge-base-en-v1.5")
    client = QdrantClient(":memory:")
    reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")
    chunks = build_chunks_recursive(filtered)
    index_chunks(chunks, COLLECTION)
    corpus_index = build_corpus_index(filtered)
    return True


# ── EXISTING PIPELINE ENTRY POINTS (unchanged) ────────────────

def run_vector(query: str) -> dict:
    """
    Vector RAG entry point.
    Routes internally between single and comparative based on
    company count — same as before adaptive RAG was added.
    """
    t0 = time.perf_counter()
    retrieved = final_retrieve(COLLECTION, query)
    answer = generate_answer(query, retrieved)
    return {
        "pipeline":    "Vector RAG",
        "answer":      answer,
        "elapsed_sec": round(time.perf_counter() - t0, 2),
        "companies":   [r.payload["company"] for r in retrieved],
        "sources": [
            {
                "company": r.payload["company"],
                "label":   f"score {float(r.score):.4f}",
                "text":    r.payload["text"],
            }
            for r in retrieved
        ],
    }


def run_vectorless(query: str) -> dict:
    """
    Vectorless RAG entry point.
    Always uses LLM navigation regardless of query type.
    Two-pass retry built in.
    """
    t0 = time.perf_counter()
    retrieved, answer, passes = vectorless_retrieve_with_retry(
        query, corpus_index, top_n=5
    )
    return {
        "pipeline":          "Vectorless RAG",
        "answer":            answer,
        "elapsed_sec":       round(time.perf_counter() - t0, 2),
        "navigation_passes": passes,
        "companies":         [r.payload["company"] for r in retrieved],
        "sources": [
            {
                "company": r.payload["company"],
                "label":   r.node_id or "section",
                "text":    r.payload["text"],
            }
            for r in retrieved
        ],
    }