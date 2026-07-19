"""
agent.py — Agentic RAG using ReAct (Reason + Act) pattern.

Imports retrieval infrastructure from pipelines.py.
Mutable globals (embedder, corpus_index, raw_cache, resolved_cache,
cache_metrics) are accessed via the module reference (pipelines.X)
rather than imported directly — this ensures agent.py always sees
the values set by init_pipelines() rather than the initial None values
that exist at import time.

Functions and constants are safe to import directly because:
  Functions: looked up at call time, always use current globals inside pipelines
  Constants: COMPANY_NAMES, COLLECTION, DONT_KNOW_PHRASE never change after load
"""

import json
import time
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import the MODULE so mutable globals are always current
import pipelines

# Import functions and constants directly — these are safe
from pipelines import (
    vectorless_retrieve_with_retry,
    detect_companies,
    classify_query,
    generate_answer,
    resolve_query,
    search_cache_store,
    save_to_cache_store,
    persist_cache_stores,
    search_episodic_memory_prod,
    save_episode_prod,
    COMPANY_NAMES,
    COLLECTION,
    DONT_KNOW_PHRASE,
)

from openai import OpenAI

llm = OpenAI()

# ── Constants ──────────────────────────────────────────────────
MAX_STEPS = 4          # fail-fast budget (was 6)
MAX_RERANK_CANDIDATES = 15
# Keep enough text that named entities / figures survive truncation
MAX_OBS_CHARS = 3500

# Soft refusals / weak answers must not poison the shared cache
_SOFT_REFUSAL_MARKERS = (
    DONT_KNOW_PHRASE,
    "i don't have enough information",
    "not enough information",
    "i was unable to find",
    "unable to find",
    "no relevant filings",
    "no information regarding",
    "i can only provide",
    "cannot answer",
)

_COMPANY_ALIASES = {
    "meta": "Meta (Facebook)",
    "facebook": "Meta (Facebook)",
    "fb": "Meta (Facebook)",
    "alphabet": "Alphabet (Google)",
    "google": "Alphabet (Google)",
    "jpmorgan": "JPMorgan Chase",
    "jp morgan": "JPMorgan Chase",
    "jpmorgan chase": "JPMorgan Chase",
    "amazon": "Amazon",
    "aws": "Amazon",
    "microsoft": "Microsoft",
    "msft": "Microsoft",
    "apple": "Apple",
    "tesla": "Tesla",
    "nvidia": "Nvidia",
    "nvda": "Nvidia",
}


def _normalize_company(company: str | None) -> str | None:
    """Map aliases / partial names to exact corpus company names."""
    if not company:
        return None
    raw = company.strip()
    if not raw:
        return None
    lower = raw.lower()
    for name in COMPANY_NAMES:
        if name.lower() == lower:
            return name
    if lower in _COMPANY_ALIASES:
        return _COMPANY_ALIASES[lower]
    # First-token / substring match against corpus names
    for name in COMPANY_NAMES:
        if name.split()[0].lower() == lower.split()[0]:
            return name
        if lower in name.lower() or name.lower() in lower:
            return name
    return raw


def _is_cacheable_answer(answer: str) -> bool:
    """Block don't-know and soft one-sided refusals from shared cache."""
    text = (answer or "").strip()
    if len(text) < 25:
        return False
    lower = text.lower()
    return not any(m in lower for m in _SOFT_REFUSAL_MARKERS)


def _answer_needs_fallback(answer: str) -> bool:
    """True when primary Agentic pass should trigger one Vectorless retry."""
    text = (answer or "").strip()
    if not text:
        return True
    lower = text.lower()
    if DONT_KNOW_PHRASE in lower:
        return True
    return not _is_cacheable_answer(text)


def _format_chunks(results) -> str:
    chunks = [
        f"[{r.payload['company']}]: {r.payload['text']}"
        for r in results
    ]
    observation = "\n\n---\n\n".join(chunks)
    if len(observation) > MAX_OBS_CHARS:
        observation = observation[:MAX_OBS_CHARS] + "... [truncated]"
    return observation


_STOPWORDS = {
    "the", "a", "an", "of", "in", "on", "for", "to", "and", "or", "which",
    "what", "how", "does", "do", "did", "was", "were", "is", "are", "its",
    "their", "with", "from", "that", "this", "these", "those", "according",
    "named", "connection", "mention", "mentions", "both", "between", "two",
    "into", "about", "over", "under", "than", "then", "also", "have", "has",
    "had", "been", "being", "as", "by", "at", "it", "be", "should", "would",
    "could", "may", "might", "per", "via", "vs", "versus", "differ", "difference",
    "nature", "look", "reader", "alongside", "own", "key", "four", "specific",
}


def _content_terms(query: str) -> list[str]:
    """Generic content tokens from a query (no domain hardcoding)."""
    terms = []
    for tok in re.findall(r"[A-Za-z0-9][A-Za-z0-9\-%]+", query):
        low = tok.lower()
        if low in _STOPWORDS or len(low) < 3:
            continue
        if low not in terms:
            terms.append(low)
    return terms


def _scroll_company_points(company: str, limit: int = 150) -> list:
    """Load up to `limit` points for one company (Agentic-local, general)."""
    from qdrant_client.models import Filter, FieldCondition, MatchValue

    points = []
    offset = None
    filt = Filter(
        must=[
            FieldCondition(key="company", match=MatchValue(value=company))
        ]
    )
    while len(points) < limit:
        batch, offset = pipelines.client.scroll(
            collection_name=COLLECTION,
            scroll_filter=filt,
            limit=min(64, limit - len(points)),
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        if not batch:
            break
        points.extend(batch)
        if offset is None:
            break
    return points


# Cleared at the start of each react_loop — avoid re-scrolling every tool call
_COMPANY_SCROLL_CACHE: dict = {}


def _scroll_company_points_cached(company: str, limit: int = 150) -> list:
    cached = _COMPANY_SCROLL_CACHE.get(company)
    if cached is not None:
        return cached
    points = _scroll_company_points(company, limit=limit)
    _COMPANY_SCROLL_CACHE[company] = points
    return points


def _lexical_candidates(points: list, query: str, top_k: int = 15) -> list:
    """Rank points by query-term overlap (generic lexical fallback)."""
    terms = _content_terms(query)
    if not terms or not points:
        return []
    scored = []
    for p in points:
        text = (p.payload.get("text") or "").lower()
        score = sum(1 for t in terms if t in text)
        if score > 0:
            scored.append((score, p))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [p for _, p in scored[:top_k]]


def _merge_points_by_id(*groups) -> list:
    seen = set()
    merged = []
    for group in groups:
        for p in group:
            pid = getattr(p, "id", None)
            key = pid if pid is not None else id(p)
            if key in seen:
                continue
            seen.add(key)
            merged.append(p)
    return merged


def _rerank_points(query: str, candidates: list, final_k: int = 5):
    if not candidates:
        return []
    pairs = [[query, c.payload["text"]] for c in candidates]
    rerank_scores = pipelines.reranker.predict(pairs)
    scored = sorted(
        zip(candidates, rerank_scores),
        key=lambda x: x[1],
        reverse=True,
    )

    class RerankedResult:
        def __init__(self, payload, score):
            self.payload = payload
            self.score = score

    return [RerankedResult(c.payload, score) for c, score in scored[:final_k]]


def _vector_search_with_rerank(
    query: str, company: str | None = None, candidate_k: int = 35, final_k: int = 5
):
    """
    Dense retrieve + (if company) cached scroll → lexical shortlist → rerank.
    Caps rerank size so multi-step Agentic stays responsive.
    """
    from qdrant_client.models import Filter, FieldCondition, MatchValue

    dense_k = 40 if company else candidate_k
    query_vector = pipelines.embedder.encode(query).tolist()
    kwargs = {
        "collection_name": COLLECTION,
        "query": query_vector,
        "limit": dense_k,
    }
    if company:
        kwargs["query_filter"] = Filter(
            must=[
                FieldCondition(
                    key="company",
                    match=MatchValue(value=company),
                )
            ]
        )
    dense = pipelines.client.query_points(**kwargs).points

    if company:
        # Lexical over dense hits only (no full-company scroll — keeps latency down)
        lexical = _lexical_candidates(dense, query, top_k=10)
        candidates = _merge_points_by_id(dense, lexical)
    else:
        lexical = _lexical_candidates(dense, query, top_k=8)
        candidates = _merge_points_by_id(dense, lexical)

    # Hard cap: cross-encoder is the expensive step
    if len(candidates) > MAX_RERANK_CANDIDATES:
        candidates = candidates[:MAX_RERANK_CANDIDATES]

    return _rerank_points(query, candidates, final_k=final_k)


# ── Tool definitions ───────────────────────────────────────────
_COMPANY_LIST = ", ".join(f"'{n}'" for n in COMPANY_NAMES)

TOOLS = [
    {
        "name": "vector_search",
        "description": (
            "Search SEC 10-K risk factor filings using semantic similarity + reranking. "
            "Best for: specific facts, named entities (suppliers, products), "
            "financial figures, and factual content questions. "
            "ALWAYS pass 'company' when the question names a company. "
            f"Exact company names: {_COMPANY_LIST}. "
            "Returns the most relevant text chunks from the filing."
        ),
        "parameters": {
            "query":   "string — what to search for",
            "company": "string (optional) — exact corpus company name"
        }
    },
    {
        "name": "vectorless_search",
        "description": (
            "Search SEC 10-K filings using LLM navigation over section structure. "
            "Best for: how sections are introduced or organized, document structure, "
            "opening statements, Summary Risk Factors vs narrative intros. "
            "Prefer this over vector_search for structural / 'how is X introduced' questions. "
            f"Exact company names: {_COMPANY_LIST}."
        ),
        "parameters": {
            "query":   "string — what to search for",
            "company": "string (optional) — exact corpus company name"
        }
    },
    {
        "name": "get_companies",
        "description": (
            "Returns the list of 8 companies available in the corpus. "
            "Use this when you need exact company names before searching."
        ),
        "parameters": {}
    },
    {
        "name": "finish",
        "description": (
            "Call when you have gathered enough search evidence. "
            "The system will synthesize the final answer from your tool results "
            "(so include thorough searches first). "
            "INPUT may be {\"answer\": \"\"} — synthesis uses observations."
        ),
        "parameters": {
            "answer": "string (optional) — ignored when search observations exist"
        }
    }
]


def _format_tools_for_prompt() -> str:
    """Formats TOOLS list into readable string for the LLM prompt."""
    lines = []
    for tool in TOOLS:
        lines.append(f"Tool: {tool['name']}")
        lines.append(f"  Description: {tool['description']}")
        if tool["parameters"]:
            for param, desc in tool["parameters"].items():
                lines.append(f"  Parameter '{param}': {desc}")
        lines.append("")
    return "\n".join(lines)


# ── Tool execution ─────────────────────────────────────────────

def _execute_vector_search(query: str, company: str = None) -> str:
    """
    Wide candidate retrieve + rerank (optionally company-filtered).
    Uses pipelines.embedder / client / reranker via module reference.
    """
    try:
        company = _normalize_company(company)
        results = _vector_search_with_rerank(
            query,
            company=company,
            candidate_k=35,
            final_k=5,
        )
        if not results:
            scope = f" in {company}'s filing" if company else ""
            return f"[No results found for '{query}'{scope}]"
        return _format_chunks(results)

    except Exception as e:
        return f"[vector_search error: {str(e)}]"


def _execute_vectorless_search(query: str, company: str = None) -> str:
    """
    Executes vectorless_search tool.
    Uses pipelines.corpus_index — always current value after init.
    """
    try:
        company = _normalize_company(company)
        if company and company in pipelines.corpus_index:
            targeted_corpus = {company: pipelines.corpus_index[company]}
        elif company:
            return (
                f"[Company '{company}' not found in corpus. "
                f"Valid names: {', '.join(COMPANY_NAMES)}]"
            )
        else:
            targeted_corpus = pipelines.corpus_index

        results, answer, passes = vectorless_retrieve_with_retry(
            query=query,
            corpus_index_map=targeted_corpus,
            top_n=5,
        )

        if not results:
            return f"[No results found for '{query}']"

        chunks = [
            f"[{r.payload['company']} - {r.node_id}]: {r.payload['text']}"
            for r in results
        ]
        observation = "\n\n---\n\n".join(chunks)
        if len(observation) > MAX_OBS_CHARS:
            observation = observation[:MAX_OBS_CHARS] + "... [truncated]"
        return observation

    except Exception as e:
        return f"[vectorless_search error: {str(e)}]"


def _execute_get_companies() -> str:
    """Returns list of available companies."""
    return "Available companies:\n" + "\n".join(
        f"  - {name}" for name in COMPANY_NAMES
    )


def execute_tool(tool_name: str, tool_input: dict) -> str:
    """
    Dispatcher — routes tool_name to the right execution function.
    Normalizes company names before search tools run.
    """
    inp = dict(tool_input or {})
    if "company" in inp:
        inp["company"] = _normalize_company(inp.get("company"))

    if tool_name == "vector_search":
        return _execute_vector_search(
            query=inp.get("query", ""),
            company=inp.get("company"),
        )
    elif tool_name == "vectorless_search":
        return _execute_vectorless_search(
            query=inp.get("query", ""),
            company=inp.get("company"),
        )
    elif tool_name == "get_companies":
        return _execute_get_companies()
    elif tool_name == "finish":
        return inp.get("answer", "")
    else:
        return f"[Unknown tool: {tool_name}]"


# ── Parallel search ────────────────────────────────────────────

def parallel_vector_search(searches: list) -> dict:
    """
    Runs multiple vector searches simultaneously.
    Cuts multi-company retrieval time roughly in half.
    """
    results = {}

    def run_one(search):
        key = search.get("company") or search.get("query", "unknown")
        obs = _execute_vector_search(
            query=search["query"],
            company=search.get("company")
        )
        return key, obs

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(run_one, s): s for s in searches}
        for future in as_completed(futures):
            try:
                key, obs = future.result()
                results[key] = obs
            except Exception as e:
                search = futures[future]
                key = search.get("company") or search.get("query", "unknown")
                results[key] = f"[Search error: {str(e)}]"

    return results


# ── Response parsing ───────────────────────────────────────────

def _parse_agent_response(response_text: str) -> dict:
    """
    Parses LLM structured response into components.
    Handles malformed responses — defaults to finish if parsing fails
    so agent never loops forever on a bad response.
    """
    thought = ""
    action = ""
    input_data = {}

    thought_match = re.search(
        r"THOUGHT:\s*(.+?)(?=ACTION:|$)", response_text, re.DOTALL
    )
    if thought_match:
        thought = thought_match.group(1).strip()

    action_match = re.search(
        r"ACTION:\s*(\w+)", response_text, re.IGNORECASE
    )
    if action_match:
        action = action_match.group(1).strip().lower()

    input_match = re.search(
        r"INPUT:\s*(\{.+?\}|\[.+?\]|.+?)(?=THOUGHT:|ACTION:|$)",
        response_text,
        re.DOTALL
    )
    if input_match:
        raw_input = input_match.group(1).strip()
        try:
            input_data = json.loads(raw_input)
        except json.JSONDecodeError:
            input_data = {"answer": raw_input}

    is_finish = (
        action == "finish"
        or not action
        or "finish" in response_text.lower()[:50]
    )

    return {
        "thought":    thought,
        "action":     action,
        "input_data": input_data,
        "is_finish":  is_finish
    }


# ── ReAct loop ─────────────────────────────────────────────────

def react_loop(query: str, episodes: list = None) -> dict:
    """
    Core ReAct loop.
    Each cycle: build prompt → call LLM → parse response
               → execute tool → add observation → repeat.
    Stops when agent calls finish or MAX_STEPS reached.
    """
    global _COMPANY_SCROLL_CACHE
    _COMPANY_SCROLL_CACHE = {}

    tool_descriptions = _format_tools_for_prompt()
    observations = []
    tool_calls_log = []
    thoughts = []

    episode_block = ""
    if episodes:
        parts = [
            f"Q: {e.get('question', '')}\nA: {e.get('answer', '')}"
            for e in episodes
        ]
        episode_block = (
            "\nRelevant past conversations:\n"
            + "\n\n".join(parts)
            + "\n"
        )

    for step in range(MAX_STEPS):

        obs_context = ""
        if observations:
            obs_context = "\n\nPrevious tool results:\n" + "\n\n".join([
                f"Tool call {i+1} ({obs['tool']}):\n{obs['result']}"
                for i, obs in enumerate(observations)
            ])

        prompt = f"""You are a research agent with access to SEC 10-K filing search tools.
Your job is to answer the user's question by searching for information
and reasoning about what you find.
{episode_block}
Available tools:
{tool_descriptions}

Exact corpus company names (use these EXACTLY in the company parameter):
{_COMPANY_LIST}

Rules:
1. Think step by step in THOUGHT before each action.
2. Always pass company= with an exact corpus name when searching one firm.
3. Factual questions: try vector_search at most once or twice, then finish.
   If evidence is weak, finish anyway — a Vectorless fallback may run after.
4. Structural questions: prefer vectorless_search.
5. Comparative questions: search EACH company once (vector or vectorless), then finish.
6. Never repeat the exact same search you already did.
7. Maximum {MAX_STEPS} tool calls — you are on step {step + 1} of {MAX_STEPS}.
   On the last step you MUST call finish.
8. Prefer don't-know over inventing facts not present in tool results.
9. Reuse distinctive nouns from the user question in search query strings.

Format your response EXACTLY like this:
THOUGHT: [your reasoning about what to do next]
ACTION: [tool name — one of: vector_search, vectorless_search, get_companies, finish]
INPUT: {{"key": "value"}}

For finish:
THOUGHT: [why the gathered evidence is enough]
ACTION: finish
INPUT: {{"answer": ""}}

User question: {query}
{obs_context}

Your response:"""

        try:
            response = llm.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=800,
            )
            response_text = response.choices[0].message.content
        except Exception as e:
            return {
                "answer":     f"Error during research: {str(e)}",
                "tool_calls": tool_calls_log,
                "cycles":     step + 1,
                "thoughts":   thoughts,
                "fallback":   None,
            }

        parsed    = _parse_agent_response(response_text)
        thought   = parsed["thought"]
        action    = parsed["action"]
        inp       = parsed["input_data"]
        is_finish = parsed["is_finish"]

        if thought:
            thoughts.append(thought)

        if is_finish or step == MAX_STEPS - 1:
            # Prefer grounded synthesis from tool evidence over free-form finish text
            if observations:
                final_answer = _synthesize_from_observations(
                    query, observations, episodes
                )
            elif is_finish and inp.get("answer"):
                final_answer = inp["answer"]
            else:
                final_answer = DONT_KNOW_PHRASE + " to answer that question."

            return {
                "answer":     final_answer,
                "tool_calls": tool_calls_log,
                "cycles":     step + 1,
                "thoughts":   thoughts,
                "fallback":   None,
            }

        tool_name        = action
        observation_text = execute_tool(tool_name, inp)

        tool_calls_log.append({
            "step":   step + 1,
            "tool":   tool_name,
            "input":  inp,
            "result": (
                observation_text[:200] + "..."
                if len(observation_text) > 200
                else observation_text
            )
        })

        observations.append({
            "tool":   tool_name,
            "input":  inp,
            "result": observation_text
        })

    return {
        "answer":     _synthesize_from_observations(query, observations, episodes),
        "tool_calls": tool_calls_log,
        "cycles":     MAX_STEPS,
        "thoughts":   thoughts,
        "fallback":   None,
    }


def _run_vectorless_fallback(
    query: str,
    episodes: list,
    agent_result: dict,
) -> dict:
    """
    One-shot Vectorless retry when the primary Agentic pass is weak/don't-know.
    Searches each detected company (max 2) or the full corpus once.
    """
    observations = []
    # Keep prior tool calls in the log for UI transparency
    tool_calls_log = list(agent_result.get("tool_calls") or [])
    thoughts = list(agent_result.get("thoughts") or [])
    thoughts.append(
        "Primary Agentic pass was low-confidence — trying Vectorless fallback once."
    )

    companies = detect_companies(query, COMPANY_NAMES)
    targets = companies[:2] if companies else [None]
    step0 = len(tool_calls_log)

    for i, company in enumerate(targets):
        obs_text = _execute_vectorless_search(query, company=company)
        entry = {
            "step": step0 + i + 1,
            "tool": "vectorless_search",
            "input": {
                "query": query,
                "company": company,
                "fallback": True,
            },
            "result": (
                obs_text[:200] + "..."
                if len(obs_text) > 200
                else obs_text
            ),
        }
        tool_calls_log.append(entry)
        observations.append({
            "tool": "vectorless_search",
            "input": entry["input"],
            "result": obs_text,
        })

    answer = _synthesize_from_observations(query, observations, episodes)
    return {
        "answer": answer,
        "tool_calls": tool_calls_log,
        "cycles": (agent_result.get("cycles") or 0) + len(targets),
        "thoughts": thoughts,
        "fallback": "vectorless",
    }


def _summarise_observations(observations: list) -> str:
    """Short summary of observations — used as error fallback."""
    if not observations:
        return "No information was found."
    parts = []
    for obs in observations:
        parts.append(f"From {obs['tool']}: {obs['result'][:200]}")
    return " | ".join(parts)


def _synthesize_from_observations(
    query: str, observations: list, episodes: list = None
) -> str:
    """
    Synthesizes final answer from accumulated search observations.
    Splits multi-chunk tool results so generate_answer sees each passage.
    """
    search_obs = [
        obs for obs in observations
        if obs.get("tool") in ("vector_search", "vectorless_search")
    ]
    if not search_obs:
        return DONT_KNOW_PHRASE + " based on the available filings."

    class ObsResult:
        def __init__(self, text, company, score):
            self.score = score
            self.payload = {"text": text, "company": company}

    fake_results = []
    score_i = 0.0
    for obs in search_obs:
        result_text = obs.get("result") or ""
        if result_text.startswith("[No results") or result_text.startswith("[Company"):
            continue
        # One observation may contain several chunks separated by ---
        parts = re.split(r"\n\n---\n\n", result_text)
        for part in parts:
            part = part.strip()
            if not part:
                continue
            company_match = re.match(r"\[([^\]]+)\]:\s*", part)
            if company_match:
                company = company_match.group(1).split(" - ")[0].strip()
                text = part[company_match.end():].strip()
            else:
                company = "Unknown"
                text = part
            if not text:
                continue
            fake_results.append(ObsResult(text, company, score_i))
            score_i += 1.0

    if not fake_results:
        return DONT_KNOW_PHRASE + " based on the available filings."

    allow_compare = len(detect_companies(query, COMPANY_NAMES)) >= 2
    # Agent-local synthesis (stronger "use concrete details" than bare finish text)
    reordered = sorted(fake_results, key=lambda r: r.score)
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
    compare_rules = ""
    if allow_compare:
        compare_rules = """
If the question asks to COMPARE companies:
- Synthesize the comparison from the company-tagged contexts.
- Use [Company A] text for company A and [Company B] text for company B.
- You do NOT need a single chunk that already compares both companies.
- Only say you don't have enough information if one side has no relevant material.
"""
    prompt = f"""Answer the question using ONLY the context below.
Do not use outside knowledge.
Include concrete details that appear in the context when they help answer
(e.g. named entities, figures, specific mechanisms or categories mentioned).
Do not invent details that are absent from the context.
If the context truly lacks the needed information, say
"I don't have enough information to answer that" rather than guessing.
{compare_rules}{episode_block}
Context:
{context}

Question: {query}

Answer:"""
    try:
        response = llm.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return response.choices[0].message.content
    except Exception:
        return generate_answer(
            query,
            fake_results,
            episodes=episodes,
            allow_compare=allow_compare,
        )


# ── Public entry point ─────────────────────────────────────────

def run_agentic(
    query: str,
    session_id: str = "default",
    chat_history=None
) -> dict:
    """
    Public entry point called by worker.py /agentic endpoint.

    Layer 1: Cache check (raw → resolved)
    Layer 2: Resolve pronouns
    Layer 3: Episodic memory recall
    Layer 4: ReAct loop
    Layer 5: Save results

    All mutable globals accessed via pipelines.X not direct import.
    """
    import numpy as np
    t0 = time.perf_counter()

    # pipelines.cache_metrics — module reference, always current
    pipelines.cache_metrics["total_queries"] += 1

    # ── Layer 1: Cache check ───────────────────────────────────
    # pipelines.raw_cache — module reference, always current
    raw_hit = search_cache_store(query, pipelines.raw_cache, "Raw cache")
    if raw_hit is not None:
        pipelines.cache_metrics["stage1_hits"] += 1
        return {
            "pipeline":          "Agentic RAG",
            "query_type":        "agentic",
            "answer":            raw_hit,
            "elapsed_sec":       round(time.perf_counter() - t0, 2),
            "cache_stage":       "raw",
            "is_confident":      True,
            "episodes_used":     0,
            "companies":         [],
            "sources":           [],
            "tool_calls":        [],
            "cycles":            0,
            "thoughts":          [],
            "navigation_passes": None,
        }

    # ── Layer 2: Resolve pronouns ──────────────────────────────
    resolved = resolve_query(query, chat_history=chat_history)

    resolved_hit = search_cache_store(
        resolved, pipelines.resolved_cache, "Resolved cache"
    )
    if resolved_hit is not None:
        pipelines.cache_metrics["stage2_hits"] += 1
        return {
            "pipeline":          "Agentic RAG",
            "query_type":        "agentic",
            "answer":            resolved_hit,
            "elapsed_sec":       round(time.perf_counter() - t0, 2),
            "cache_stage":       "resolved",
            "is_confident":      True,
            "episodes_used":     0,
            "companies":         [],
            "sources":           [],
            "tool_calls":        [],
            "cycles":            0,
            "thoughts":          [],
            "navigation_passes": None,
        }

    # ── Layer 3: Episodic memory recall ────────────────────────
    pipelines.cache_metrics["full_pipeline_runs"] += 1
    episodes = search_episodic_memory_prod(
        resolved, top_k=2, exclude_session=session_id
    )

    # Skip episodic for comparative / multi-company (same as Adaptive)
    query_type_preview = classify_query(resolved)
    is_multi = len(detect_companies(resolved, COMPANY_NAMES)) >= 2
    episodes_for_generation = (
        [] if (query_type_preview == "comparative" or is_multi) else episodes
    )

    # ── Layer 4: ReAct loop ────────────────────────────────────
    agent_result = react_loop(resolved, episodes=episodes_for_generation)

    # Fail-fast: if primary pass is don't-know / soft refusal → one Vectorless try
    fallback_used = None
    if _answer_needs_fallback(agent_result.get("answer") or ""):
        agent_result = _run_vectorless_fallback(
            resolved, episodes_for_generation, agent_result
        )
        fallback_used = agent_result.get("fallback")

    answer     = agent_result["answer"]
    tool_calls = agent_result["tool_calls"]
    cycles     = agent_result["cycles"]
    thoughts   = agent_result["thoughts"]

    companies_searched = list({
        tc["input"].get("company")
        for tc in tool_calls
        if tc["input"].get("company")
    })

    sources = []
    for tc in tool_calls:
        if tc["tool"] in ("vector_search", "vectorless_search"):
            sources.append({
                "company": tc["input"].get("company", "Multiple"),
                "label":   (
                    f"{tc['tool']} (step {tc['step']}"
                    + (", fallback)" if (tc.get("input") or {}).get("fallback") else ")")
                ),
                "text":    tc["result"]
            })

    # ── Layer 5: Save results ──────────────────────────────────
    # Only save confident, non-soft-refusal answers — never poison shared cache
    is_confident = DONT_KNOW_PHRASE not in answer.lower()
    if is_confident and _is_cacheable_answer(answer):
        qvec = pipelines.embedder.encode(query).tolist()
        rvec = pipelines.embedder.encode(resolved).tolist()
        save_to_cache_store(query, qvec, answer, pipelines.raw_cache)
        save_to_cache_store(resolved, rvec, answer, pipelines.resolved_cache)
        persist_cache_stores()
        save_episode_prod(resolved, answer, session_id)
    elif is_confident and not _is_cacheable_answer(answer):
        print(
            "  Agentic: blocked cache save (soft refusal / weak answer): "
            f"{answer[:80]}..."
        )

    return {
        "pipeline":          "Agentic RAG",
        "query_type":        "agentic",
        "answer":            answer,
        "elapsed_sec":       round(time.perf_counter() - t0, 2),
        "cache_stage":       "miss",
        "is_confident":      is_confident,
        "episodes_used":     len(episodes_for_generation),
        "companies":         companies_searched,
        "sources":           sources,
        "tool_calls":        tool_calls,
        "cycles":            cycles,
        "thoughts":          thoughts,
        "navigation_passes": None,
        "resolved_query":    resolved,
        "fallback":          fallback_used,
        "fallback_used":     bool(fallback_used),
    }