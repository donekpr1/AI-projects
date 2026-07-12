"""RAG pipelines — Adaptive/Vector/Vectorless + disk Qdrant + cache + episodic memory."""
import re
import json
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.linalg import norm
from dotenv import load_dotenv
from openai import OpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance, PointStruct,
    Filter, FieldCondition, MatchValue,
)
from sentence_transformers import SentenceTransformer, CrossEncoder

ROOT = Path(__file__).parent
COLLECTION = "risk_factors_recursive"
EPISODIC_COLLECTION = "episodic_memory"
QDRANT_DIR = ROOT / "qdrant_data"
CORPUS_INDEX_PATH = ROOT / "corpus_index.json"
CACHE_PATH = ROOT / "cache_store.json"

DONT_KNOW_PHRASE = "i don't have enough information"
VALID_QUERY_TYPES = ["simple_factual", "comparative", "structural", "negative"]

CACHE_SIMILARITY_THRESHOLD = 0.85
TOPIC_MATCH_THRESHOLD = 0.5
EPISODIC_SIMILARITY_THRESHOLD = 0.75
EPISODIC_DEDUP_THRESHOLD = 0.95
RETENTION_DAYS = 90

TOPIC_GROUPS = {
    "supplier":      ["supplier", "supply", "vendor", "partner", "panasonic",
                      "sourcing", "procurement", "third-party"],
    "manufacturing": ["manufactur", "factory", "gigafactory", "production",
                      "assembly", "plant", "facility", "operations"],
    "revenue":       ["revenue", "sales", "advertising", "income", "earnings",
                      "profit", "financial", "monetiz"],
    "competition":   ["competi", "rival", "market share", "competitor",
                      "competitive", "industry"],
    "covid":         ["covid", "pandemic", "coronavirus", "health", "lockdown"],
    "regulatory":    ["regulat", "compliance", "legal", "law", "government",
                      "sec", "filing", "policy"],
    "technology":    ["gpu", "chip", "semiconductor", "software", "cloud",
                      "platform", "computing", "data center"],
    "risk":          ["risk", "uncertainty", "challenge", "threat", "exposure"],
    "structure":     ["introduce", "structure", "organized", "section", "open"],
}

load_dotenv()
llm = OpenAI()

filtered = pd.read_parquet(ROOT / "filtered_2020_filings.parquet")
COMPANY_NAMES = filtered["company_name"].unique().tolist()

embedder = None
client = None
reranker = None
corpus_index = None

raw_cache = []
resolved_cache = []
cache_metrics = {
    "total_queries": 0,
    "stage1_hits": 0,
    "stage2_hits": 0,
    "full_pipeline_runs": 0,
    "topic_mismatches": 0,
}


# ── basics ────────────────────────────────────────────────────

def detect_companies(query, company_names):
    return [
        name for name in company_names
        if name.split()[0].lower() in query.lower()
    ]


def generate_answer(query, retrieved_chunks, episodes=None):
    """
    Builds prompt from retrieved chunks + optional episodic context.
    Works for both Qdrant ScoredPoint and VectorlessResult objects
    because both expose .score and .payload["text"]/["company"].
    Reorders chunks by score (compaction) before building prompt.
    Episodes injected as separate labeled section if provided.
    """
    reordered = sorted(retrieved_chunks, key=lambda r: r.score)
    context = "\n\n---\n\n".join(
        f"[{r.payload['company']}]: {r.payload['text']}"
        for r in reordered
    )
    episode_block = ""
    if episodes:
        parts = [
            f"Q: {e.get('question', '')}\nA: {e.get('answer', '')}"
            for e in episodes
        ]
        episode_block = (
            "\n\nRelevant past conversations (episodic memory):\n"
            + "\n\n".join(parts) + "\n"
        )
    prompt = f"""Answer the question using ONLY the context below.
If the context doesn't contain enough information to answer,
say "I don't have enough information to answer that" rather than guessing.
{episode_block}
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
            size=len(embeddings[0]), distance=Distance.COSINE
        ),
    )
    points = [
        PointStruct(id=i, vector=embeddings[i].tolist(), payload=chunks[i])
        for i in range(len(chunks))
    ]
    client.upsert(collection_name=collection_name, points=points)
    return len(points)


def ensure_vector_index():
    """
    Creates and populates the Qdrant vector collection only if it
    doesn't already exist or is empty. Persistent Qdrant means this
    only runs once — subsequent startups reuse the existing index.
    """
    global client
    if client.collection_exists(COLLECTION):
        n = client.count(collection_name=COLLECTION).count
        if n > 0:
            print(f"Using existing Qdrant collection '{COLLECTION}' ({n} points)")
            return n
    print("Building vector index (one-time)...")
    chunks = build_chunks_recursive(filtered)
    return index_chunks(chunks, COLLECTION)


def load_or_build_corpus_index():
    """
    Loads corpus_index from JSON if it exists, otherwise builds and
    saves it. Corpus index is the Python dict used by vectorless RAG.
    Persisting avoids re-splitting all 8 company filings on every restart.
    """
    if CORPUS_INDEX_PATH.exists() and CORPUS_INDEX_PATH.stat().st_size > 0:
        print(f"Loading corpus_index from {CORPUS_INDEX_PATH.name}")
        with open(CORPUS_INDEX_PATH, encoding="utf-8") as f:
            return json.load(f)
    print("Building corpus_index (one-time)...")
    index = build_corpus_index(filtered)
    with open(CORPUS_INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(index, f)
    return index


# ── vector retrieval ──────────────────────────────────────────

def rerank_retrieve(collection_name, query, candidate_k=35, final_k=5):
    """
    Single-company path: dense search (35 candidates) →
    cross-encoder reranking → top 5.
    candidate_k=35 because Panasonic chunk ranked 29th in dense
    search — wider pool gives reranker chance to surface buried chunks.
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
        key=lambda x: x[1], reverse=True
    )

    class RerankedResult:
        def __init__(self, payload, score):
            self.payload = payload
            self.score = score

    return [RerankedResult(c.payload, score) for c, score in scored[:final_k]]


def retrieve_with_decomposition(collection_name, query, top_k=3):
    """
    Comparative path: separate per-company metadata-filtered search.
    Returns top_k chunks per company — guarantees equal representation.
    Fixes: single global search let one company sweep all top-k slots.
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
                        key="company", match=MatchValue(value=company)
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
    """Original vector router — kept for _core_vector path."""
    if len(detect_companies(query, COMPANY_NAMES)) >= 2:
        return retrieve_with_decomposition(collection_name, query, top_k=3)
    return rerank_retrieve(collection_name, query, candidate_k, final_k)


# ── vectorless ────────────────────────────────────────────────

def split_risk_factors_into_nodes(
    section_text, company, min_chars=80, max_chars=1500
):
    if not section_text or not str(section_text).strip():
        return [{
            "id": f"{company}::0", "company": company,
            "title": "[EMPTY SECTION]", "text": "", "char_len": 0,
        }]
    text = str(section_text).strip()
    raw_parts = re.split(
        r'\n(?=\s*(?:Item 1A\.?|ITEM 1A\.?|RISK FACTORS|•|\(\d+\)|\d+\.\s+[A-Z]))',
        text, flags=re.IGNORECASE,
    )
    nodes, buf = [], ""

    def flush_buffer():
        nonlocal buf
        chunk = buf.strip()
        buf = ""
        if len(chunk) < min_chars:
            return
        nodes.append({
            "id": f"{company}::{len(nodes)}", "company": company,
            "title": chunk[:120].replace("\n", " ").strip(),
            "text": chunk, "char_len": len(chunk),
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
                        "id": f"{company}::{len(nodes)}", "company": company,
                        "title": sub[:120].replace("\n", " ").strip(),
                        "text": sub, "char_len": len(sub),
                    })
        else:
            if len(buf) + len(part) + 1 > max_chars:
                flush_buffer()
            buf = f"{buf}\n{part}".strip() if buf else part
    if buf:
        flush_buffer()
    if not nodes:
        nodes.append({
            "id": f"{company}::0", "company": company,
            "title": text[:120].replace("\n", " "),
            "text": text, "char_len": len(text),
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
    start, end = raw_text.find("["), raw_text.rfind("]")
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
            score=0.0, node_id=nodes[0]["id"],
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
    results, seen = [], set()
    for rank, nid in enumerate(selected_ids):
        if nid in seen:
            continue
        node = node_map.get(nid)
        if node and node["char_len"] > 0:
            seen.add(nid)
            results.append(VectorlessResult(
                payload={"company": company, "text": node["text"]},
                score=float(rank), node_id=nid,
            ))
        if len(results) >= top_n:
            break
    if not results and nodes[0]["char_len"] > 0:
        results.append(VectorlessResult(
            payload={"company": company, "text": nodes[0]["text"]},
            score=0.0, node_id=nodes[0]["id"],
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


def vectorless_retrieve_with_retry(query, corpus_index_map, top_n=5, episodes=None):
    """
    Pass 1: navigate + generate.
    Pass 2: if low confidence, exclude already-tried nodes and navigate again.
    Second pass results get score -10 (lower = higher priority in merged list).
    episodes: optional past Q&A pairs injected into generation prompt.
    """
    first_results = vectorless_retrieve(query, corpus_index_map, top_n=top_n)
    first_answer = generate_answer(query, first_results, episodes=episodes)
    if DONT_KNOW_PHRASE not in first_answer.lower():
        return first_results, first_answer, 1

    companies = detect_companies(query, COMPANY_NAMES)
    target = companies if companies else list(corpus_index_map.keys())
    tried_ids = {r.node_id for r in first_results if r.node_id}
    second_results = []
    for company in target:
        nodes = corpus_index_map.get(company, [])
        second_results.extend(navigate_company_tree(
            query, company, nodes,
            top_n=top_n, exclude_ids=tried_ids, pass_label="second",
        ))
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
    second_answer = generate_answer(query, final_results, episodes=episodes)
    return final_results, second_answer, 2


# ── cache ─────────────────────────────────────────────────────

def extract_topics(text: str) -> set:
    text_lower = text.lower()
    topics = set()
    for topic_label, keywords in TOPIC_GROUPS.items():
        for keyword in keywords:
            if keyword in text_lower:
                topics.add(topic_label)
                break
    return topics


def topics_match(query1: str, query2: str) -> bool:
    topics1, topics2 = extract_topics(query1), extract_topics(query2)
    if not topics1 or not topics2:
        return True
    jaccard = len(topics1 & topics2) / len(topics1 | topics2)
    return jaccard >= TOPIC_MATCH_THRESHOLD


def search_cache_store(query, cache_store, label="cache"):
    """
    Two-gate cache lookup:
      Gate 1: cosine similarity >= CACHE_SIMILARITY_THRESHOLD (0.85)
      Gate 2: topics_match() — Jaccard >= TOPIC_MATCH_THRESHOLD (0.5)
    Both must pass for cache hit.
    topic_mismatches tracked for monitoring — rising rate signals
    the threshold may need tuning.
    """
    if not cache_store:
        return None
    query_vector = embedder.encode(query)
    best_score, best_entry = 0.0, None
    for entry in cache_store:
        cached_vector = np.array(entry["query_vector"])
        similarity = float(
            np.dot(query_vector, cached_vector)
            / (norm(query_vector) * norm(cached_vector))
        )
        if similarity > best_score:
            best_score, best_entry = similarity, entry
    if best_score < CACHE_SIMILARITY_THRESHOLD:
        return None
    if not topics_match(query, best_entry["query"]):
        cache_metrics["topic_mismatches"] += 1
        return None
    best_entry["hit_count"] = best_entry.get("hit_count", 0) + 1
    return best_entry["answer"]


def save_to_cache_store(query, query_vector, answer, cache_store, label="cache"):
    """
    Saves Q&A to cache with dedup check.
    If very similar query already exists (similarity > 0.95),
    updates it rather than adding a duplicate.
    """
    qv = np.array(query_vector)
    for entry in cache_store:
        cached_vector = np.array(entry["query_vector"])
        similarity = float(
            np.dot(qv, cached_vector) / (norm(qv) * norm(cached_vector))
        )
        if similarity > 0.95:
            entry["answer"] = answer
            entry["timestamp"] = datetime.now().isoformat()
            return
    cache_store.append({
        "query_vector": qv.tolist(),
        "query": query,
        "answer": answer,
        "timestamp": datetime.now().isoformat(),
        "hit_count": 0,
    })


def load_cache_stores():
    """Loads raw and resolved caches from disk on startup."""
    global raw_cache, resolved_cache
    if CACHE_PATH.exists() and CACHE_PATH.stat().st_size > 0:
        with open(CACHE_PATH, encoding="utf-8") as f:
            data = json.load(f)
        raw_cache = data.get("raw_cache", [])
        resolved_cache = data.get("resolved_cache", [])
        print(f"Loaded cache: raw={len(raw_cache)} resolved={len(resolved_cache)}")
    else:
        raw_cache, resolved_cache = [], []


def persist_cache_stores():
    """Saves raw and resolved caches to disk after every full pipeline run."""
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {"raw_cache": raw_cache, "resolved_cache": resolved_cache}, f
        )


def resolve_query(query, chat_history=None):
    """
    Rewrites query as standalone using recent chat history.
    Resolves pronouns: "their" → company name, "that" → topic.
    Returns query unchanged if no history or LLM call fails.
    max_tokens=120 — rewritten question should be short.
    """
    if not chat_history:
        return query
    hist = "\n".join(
        f"{m['role']}: {m['content'][:300]}"
        for m in chat_history[-4:]
    )
    prompt = f"""Rewrite the latest user question as a standalone question
using the chat history if needed.
Return ONLY the rewritten question.

Chat history:
{hist}

Latest question: {query}
"""
    try:
        response = llm.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=120,
        )
        return response.choices[0].message.content.strip() or query
    except Exception:
        return query


# ── episodic memory ───────────────────────────────────────────

def setup_episodic_collection():
    global client
    names = [c.name for c in client.get_collections().collections]
    if EPISODIC_COLLECTION not in names:
        client.create_collection(
            collection_name=EPISODIC_COLLECTION,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE),
        )
        print(f"Created episodic collection: {EPISODIC_COLLECTION}")


def apply_retention_policy():
    """
    Deletes episodes older than RETENTION_DAYS from Qdrant.
    Called at startup — in production this would run as a scheduled
    daily job rather than at startup, but startup is fine for POC.
    """
    cutoff = (datetime.now() - timedelta(days=RETENTION_DAYS)).isoformat()
    results, _ = client.scroll(
        collection_name=EPISODIC_COLLECTION,
        limit=1000, with_payload=True, with_vectors=False
    )
    old_ids = [
        r.id for r in results
        if r.payload.get("timestamp", "") < cutoff
    ]
    if old_ids:
        client.delete(
            collection_name=EPISODIC_COLLECTION,
            points_selector=old_ids
        )
        print(f"Retention: deleted {len(old_ids)} old episodes")


def save_episode_prod(question, answer, session_id):
    """
    Saves Q&A to Qdrant episodic_memory with dedup.
    Skips saving if answer is low confidence — don't store
    'I don't know' answers as useful past context.
    Dedup: if very similar question exists (score > 0.95),
    updates it rather than creating a duplicate.
    """
    if DONT_KNOW_PHRASE in answer.lower():
        return
    question_vector = embedder.encode(question).tolist()
    existing = client.query_points(
        collection_name=EPISODIC_COLLECTION,
        query=question_vector,
        limit=1,
        score_threshold=EPISODIC_DEDUP_THRESHOLD,
    )
    point_id = existing.points[0].id if existing.points else str(uuid.uuid4())
    client.upsert(
        collection_name=EPISODIC_COLLECTION,
        points=[PointStruct(
            id=point_id,
            vector=question_vector,
            payload={
                "question": question,
                "answer": answer,
                "session_id": session_id,
                "timestamp": datetime.now().isoformat(),
            },
        )],
    )


def search_episodic_memory_prod(query, top_k=2, exclude_session=None):
    """
    Searches Qdrant episodic_memory for past Q&A relevant to query.
    Threshold: 0.75 — more permissive than cache (0.85) because
    partial episode relevance still enriches the prompt.
    Session exclusion prevents self-recall within same session.
    """
    query_vector = embedder.encode(query).tolist()
    query_filter = None
    if exclude_session:
        query_filter = Filter(
            must_not=[FieldCondition(
                key="session_id", match=MatchValue(value=exclude_session)
            )]
        )
    results = client.query_points(
        collection_name=EPISODIC_COLLECTION,
        query=query_vector,
        query_filter=query_filter,
        score_threshold=EPISODIC_SIMILARITY_THRESHOLD,
        limit=top_k,
    )
    return [r.payload for r in results.points]


# ── adaptive ──────────────────────────────────────────────────

def classify_query(query: str) -> str:
    """
    Classifies query into one of four types using GPT-4o-mini.
    Based ONLY on query surface language — never checks corpus data.
    Conservative: defaults to simple_factual when uncertain so
    valid questions are never wrongly skipped.
    max_tokens=10 — one word response, minimal cost and latency.
    Fallback: any unexpected response → simple_factual (safe default).
    Priority order enforced in prompt: structural > comparative > simple_factual > negative.
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
            return "simple_factual"
        return classification
    except Exception:
        return "simple_factual"


def _reformulate(query: str) -> str:
    """
    Broadens a failed query to better match filing vocabulary.
    Called only when first retrieval produced low confidence answer.
    temperature=0.3 adds variation so retry doesn't repeat failed phrasing.
    """
    prompt = f"""This question failed to retrieve a good answer from SEC 10-K filings:
{query}

Rephrase it to be broader and more likely to match financial filing language.
Return ONLY the rephrased question."""
    try:
        response = llm.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=120,
        )
        return response.choices[0].message.content.strip() or query
    except Exception:
        return query


def _core_adaptive(query: str, episodes=None) -> dict:
    """
    Core adaptive routing — classify query then choose retrieval strategy.

    Routing:
      simple_factual → vector (dense + rerank) — proven on Q2-Q5
      comparative    → vector (per-company filter) — proven on Q6
      structural     → vectorless (LLM navigation) — proven on Q7
      negative       → vectorless safety net (corpus check before skip)

    One retry with reformulation for simple_factual and comparative
    when first attempt produces low confidence answer.
    No retry for structural/negative — vectorless_retrieve_with_retry
    handles its own two-pass retry internally.
    """
    query_type = classify_query(query)
    navigation_passes = None

    if query_type == "simple_factual":
        retrieved = rerank_retrieve(COLLECTION, query)
        answer = generate_answer(query, retrieved, episodes=episodes)
        # Retry once with reformulated query if low confidence
        if DONT_KNOW_PHRASE in answer.lower():
            reformulated = _reformulate(query)
            retrieved = rerank_retrieve(COLLECTION, reformulated)
            answer = generate_answer(reformulated, retrieved, episodes=episodes)
        sources = [
            {
                "company": r.payload["company"],
                "label": f"score {float(r.score):.4f}",
                "text": r.payload["text"],
            }
            for r in retrieved
        ]

    elif query_type == "comparative":
        retrieved = retrieve_with_decomposition(COLLECTION, query, top_k=3)
        answer = generate_answer(query, retrieved, episodes=episodes)
        # Retry once with reformulated query if low confidence
        if DONT_KNOW_PHRASE in answer.lower():
            reformulated = _reformulate(query)
            retrieved = retrieve_with_decomposition(
                COLLECTION, reformulated, top_k=3
            )
            answer = generate_answer(reformulated, retrieved, episodes=episodes)
        sources = [
            {
                "company": r.payload["company"],
                "label": f"score {float(r.score):.4f}",
                "text": r.payload["text"],
            }
            for r in retrieved
        ]

    elif query_type == "structural":
        # Vectorless handles its own two-pass retry internally
        retrieved, answer, navigation_passes = vectorless_retrieve_with_retry(
            query, corpus_index, top_n=5, episodes=episodes
        )
        sources = [
            {
                "company": r.payload["company"],
                "label": r.node_id or "section",
                "text": r.payload["text"],
            }
            for r in retrieved
        ]

    else:
        # negative — vectorless safety net
        # Classifier said negative but never checked corpus.
        # Give corpus a chance to override the classification.
        # top_n=3 (smaller — low confidence this will work).
        retrieved, answer, navigation_passes = vectorless_retrieve_with_retry(
            query, corpus_index, top_n=3, episodes=episodes
        )
        sources = [
            {
                "company": r.payload["company"],
                "label": r.node_id or "section",
                "text": r.payload["text"],
            }
            for r in retrieved
        ]

    return {
        "pipeline":          "Adaptive RAG",
        "query_type":        query_type,
        "navigation_passes": navigation_passes,
        "answer":            answer,
        "companies":         list({s["company"] for s in sources}),
        "sources":           sources,
    }


def _core_vector(query: str, episodes=None) -> dict:
    """
    Vector RAG core — routes internally between single and comparative.
    One retry with reformulation on low confidence.
    """
    retrieved = final_retrieve(COLLECTION, query)
    answer = generate_answer(query, retrieved, episodes=episodes)
    if DONT_KNOW_PHRASE in answer.lower():
        reformulated = _reformulate(query)
        retrieved = final_retrieve(COLLECTION, reformulated)
        answer = generate_answer(reformulated, retrieved, episodes=episodes)
    return {
        "pipeline":          "Vector RAG",
        "query_type":        None,
        "navigation_passes": None,
        "answer":            answer,
        "companies":         [r.payload["company"] for r in retrieved],
        "sources": [
            {
                "company": r.payload["company"],
                "label":   f"score {float(r.score):.4f}",
                "text":    r.payload["text"],
            }
            for r in retrieved
        ],
    }


def _core_vectorless(query: str, episodes=None) -> dict:
    """
    Vectorless RAG core — always uses LLM navigation.
    Two-pass retry handled internally by vectorless_retrieve_with_retry.
    """
    retrieved, answer, passes = vectorless_retrieve_with_retry(
        query, corpus_index, top_n=5, episodes=episodes
    )
    return {
        "pipeline":          "Vectorless RAG",
        "query_type":        None,
        "navigation_passes": passes,
        "answer":            answer,
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


# ── unified entry point ───────────────────────────────────────

def run_query(
    query: str,
    pipeline: str = "Adaptive RAG",
    session_id: str = "default",
    chat_history=None
) -> dict:
    """
    Unified entry point for all three pipelines.

    Layer 1 — raw cache check:
      Checks raw query against raw_cache.
      Two gates: similarity >= 0.85 AND topic match.
      HIT → return instantly (0.03s, zero LLM calls).

    Layer 2 — resolve + resolved cache check:
      Resolves pronouns using chat_history.
      Checks resolved query against resolved_cache.
      HIT → return (one LLM call for resolution, no retrieval).

    Layer 3 — episodic memory recall:
      Searches Qdrant episodic_memory for relevant past Q&A.
      Found episodes injected into generation prompt as context.

    Layer 4 — core pipeline:
      Routes to _core_adaptive, _core_vector, or _core_vectorless.
      Each handles its own retrieval strategy and optional retry.

    Layer 5 — save (only on confident answers):
      Saves to raw_cache and resolved_cache (persisted to disk).
      Saves to Qdrant episodic_memory.
      Never saves low confidence answers to any store.
    """
    global raw_cache, resolved_cache
    t0 = time.perf_counter()
    cache_metrics["total_queries"] += 1

    # Layer 1 — raw cache
    raw_hit = search_cache_store(query, raw_cache, "Raw cache")
    if raw_hit is not None:
        cache_metrics["stage1_hits"] += 1
        return {
            "pipeline":          pipeline,
            "query_type":        None,
            "navigation_passes": None,
            "answer":            raw_hit,
            "elapsed_sec":       round(time.perf_counter() - t0, 2),
            "cache_stage":       "raw",
            "is_confident":      True,
            "episodes_used":     0,
            "companies":         [],
            "sources":           [],
        }

    # Layer 2 — resolve + resolved cache
    resolved = resolve_query(query, chat_history=chat_history)
    resolved_hit = search_cache_store(resolved, resolved_cache, "Resolved cache")
    if resolved_hit is not None:
        cache_metrics["stage2_hits"] += 1
        return {
            "pipeline":          pipeline,
            "query_type":        None,
            "navigation_passes": None,
            "answer":            resolved_hit,
            "elapsed_sec":       round(time.perf_counter() - t0, 2),
            "cache_stage":       "resolved",
            "is_confident":      True,
            "episodes_used":     0,
            "companies":         [],
            "sources":           [],
        }

    # Layer 3 — episodic recall
    cache_metrics["full_pipeline_runs"] += 1
    episodes = search_episodic_memory_prod(resolved, top_k=2)

    # Layer 4 — core pipeline
    if pipeline == "Vector RAG":
        result = _core_vector(resolved, episodes=episodes)
    elif pipeline == "Vectorless RAG":
        result = _core_vectorless(resolved, episodes=episodes)
    else:
        result = _core_adaptive(resolved, episodes=episodes)

    # Layer 5 — save only confident answers
    # Never cache or store "I don't know" answers
    is_confident = DONT_KNOW_PHRASE not in result["answer"].lower()
    if is_confident:
        qvec = embedder.encode(query).tolist()
        rvec = embedder.encode(resolved).tolist()
        save_to_cache_store(query, qvec, result["answer"], raw_cache)
        save_to_cache_store(resolved, rvec, result["answer"], resolved_cache)
        persist_cache_stores()
        save_episode_prod(resolved, result["answer"], session_id)

    result["elapsed_sec"]   = round(time.perf_counter() - t0, 2)
    result["cache_stage"]   = "miss"
    result["is_confident"]  = is_confident
    result["episodes_used"] = len(episodes)
    result["resolved_query"] = resolved
    return result


# ── public API ────────────────────────────────────────────────

def run_adaptive(
    query: str, session_id: str = "default", chat_history=None
) -> dict:
    return run_query(
        query, "Adaptive RAG",
        session_id=session_id, chat_history=chat_history
    )


def run_vector(
    query: str, session_id: str = "default", chat_history=None
) -> dict:
    return run_query(
        query, "Vector RAG",
        session_id=session_id, chat_history=chat_history
    )


def run_vectorless(
    query: str, session_id: str = "default", chat_history=None
) -> dict:
    return run_query(
        query, "Vectorless RAG",
        session_id=session_id, chat_history=chat_history
    )


# ── init ──────────────────────────────────────────────────────

def init_pipelines():
    """
    Called once at Streamlit startup.
    Order matters:
      1. embedder — needed by all embedding operations
      2. client (persistent Qdrant) — survives restarts
      3. reranker — needed by rerank_retrieve
      4. ensure_vector_index — builds only if missing
      5. corpus_index — loads from JSON or builds once
      6. episodic collection — creates if missing
      7. apply_retention_policy — removes old episodes
      8. load_cache_stores — loads cache from disk
    """
    global embedder, client, reranker, corpus_index
    embedder = SentenceTransformer("BAAI/bge-base-en-v1.5")
    QDRANT_DIR.mkdir(exist_ok=True)
    client = QdrantClient(path=str(QDRANT_DIR))
    reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")
    ensure_vector_index()
    corpus_index = load_or_build_corpus_index()
    setup_episodic_collection()
    apply_retention_policy()
    load_cache_stores()
    return True