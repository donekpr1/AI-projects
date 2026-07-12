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
    rerank_retrieve,
    retrieve_with_decomposition,
    vectorless_retrieve_with_retry,
    detect_companies,
    generate_answer,
    final_retrieve,
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
MAX_STEPS = 6
MAX_OBS_CHARS = 800


# ── Tool definitions ───────────────────────────────────────────
TOOLS = [
    {
        "name": "vector_search",
        "description": (
            "Search SEC 10-K risk factor filings using semantic similarity. "
            "Best for: specific facts, named entities (company names, products, people), "
            "financial figures, and factual content questions. "
            "Use the optional 'company' parameter to restrict search to one company — "
            "always use this when you know which company to search. "
            "Returns the most relevant text chunks from the filing."
        ),
        "parameters": {
            "query":   "string — what to search for",
            "company": "string (optional) — restrict to this company name exactly"
        }
    },
    {
        "name": "vectorless_search",
        "description": (
            "Search SEC 10-K filings using LLM navigation over section structure. "
            "Best for: how sections are introduced or organized, document structure "
            "questions, opening statements, and section-level content. "
            "Use when vector_search returns content but misses the structural context. "
            "Use the optional 'company' parameter to restrict to one company."
        ),
        "parameters": {
            "query":   "string — what to search for",
            "company": "string (optional) — restrict to this company name exactly"
        }
    },
    {
        "name": "get_companies",
        "description": (
            "Returns the list of 8 companies available in the corpus. "
            "Use this when you need to know which companies can be searched, "
            "or when the question asks about 'all companies' without naming them."
        ),
        "parameters": {}
    },
    {
        "name": "finish",
        "description": (
            "Call this when you have gathered enough information to answer "
            "the question completely. Pass your full synthesized answer. "
            "Call finish as soon as you have enough — do not over-search."
        ),
        "parameters": {
            "answer": "string — your complete final answer to the user's question"
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
    Executes vector_search tool.
    Uses pipelines.embedder (not imported embedder) because
    embedder is None at import time and only set after init_pipelines().
    Uses pipelines.client for the same reason.
    """
    try:
        if company:
            from qdrant_client.models import Filter, FieldCondition, MatchValue

            # pipelines.embedder — always current value, not import-time None
            query_vector = pipelines.embedder.encode(query).tolist()

            # pipelines.client — always current Qdrant client
            raw_results = pipelines.client.query_points(
                collection_name=COLLECTION,
                query=query_vector,
                query_filter=Filter(
                    must=[FieldCondition(
                        key="company",
                        match=MatchValue(value=company)
                    )]
                ),
                limit=5
            ).points

            if not raw_results:
                return f"[No results found for '{query}' in {company}'s filing]"

            chunks = [
                f"[{r.payload['company']}]: {r.payload['text']}"
                for r in raw_results
            ]
        else:
            results = rerank_retrieve(COLLECTION, query)
            if not results:
                return f"[No results found for '{query}']"
            chunks = [
                f"[{r.payload['company']}]: {r.payload['text']}"
                for r in results
            ]

        observation = "\n\n---\n\n".join(chunks)
        if len(observation) > MAX_OBS_CHARS:
            observation = observation[:MAX_OBS_CHARS] + "... [truncated]"
        return observation

    except Exception as e:
        return f"[vector_search error: {str(e)}]"


def _execute_vectorless_search(query: str, company: str = None) -> str:
    """
    Executes vectorless_search tool.
    Uses pipelines.corpus_index — always current value after init.
    """
    try:
        if company and company in pipelines.corpus_index:
            # pipelines.corpus_index — not import-time None
            targeted_corpus = {company: pipelines.corpus_index[company]}
        elif company:
            return f"[Company '{company}' not found in corpus]"
        else:
            targeted_corpus = pipelines.corpus_index

        results, answer, passes = vectorless_retrieve_with_retry(
            query=query,
            corpus_index_map=targeted_corpus,
            top_n=3
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
    Returns observation string added to agent context.
    """
    if tool_name == "vector_search":
        return _execute_vector_search(
            query=tool_input.get("query", ""),
            company=tool_input.get("company", None)
        )
    elif tool_name == "vectorless_search":
        return _execute_vectorless_search(
            query=tool_input.get("query", ""),
            company=tool_input.get("company", None)
        )
    elif tool_name == "get_companies":
        return _execute_get_companies()
    elif tool_name == "finish":
        return tool_input.get("answer", "")
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

Rules:
1. Think step by step in THOUGHT before each action.
2. After each tool result, decide if you have enough information.
3. Call finish() as soon as you have a complete answer.
4. Maximum {MAX_STEPS} tool calls — you are on step {step + 1} of {MAX_STEPS}.
5. If on the last step, you MUST call finish with whatever you have.
6. Always use the 'company' parameter when you know which company to search.
7. Never repeat the exact same search you already did.

Format your response EXACTLY like this:
THOUGHT: [your reasoning about what to do next]
ACTION: [tool name — one of: vector_search, vectorless_search, get_companies, finish]
INPUT: {{"key": "value"}}

For finish:
THOUGHT: [your reasoning that you have enough information]
ACTION: finish
INPUT: {{"answer": "your complete answer here"}}

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
                "thoughts":   thoughts
            }

        parsed    = _parse_agent_response(response_text)
        thought   = parsed["thought"]
        action    = parsed["action"]
        inp       = parsed["input_data"]
        is_finish = parsed["is_finish"]

        if thought:
            thoughts.append(thought)

        if is_finish or step == MAX_STEPS - 1:
            if is_finish and "answer" in inp:
                final_answer = inp["answer"]
            elif observations:
                final_answer = _synthesize_from_observations(
                    query, observations, episodes
                )
            else:
                final_answer = DONT_KNOW_PHRASE + " to answer that question."

            return {
                "answer":     final_answer,
                "tool_calls": tool_calls_log,
                "cycles":     step + 1,
                "thoughts":   thoughts
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
        "thoughts":   thoughts
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
    Synthesizes final answer from accumulated observations
    when agent runs out of steps without calling finish.
    Builds fake result objects that generate_answer() can consume.
    """
    if not observations:
        return DONT_KNOW_PHRASE + " based on the available filings."

    class ObsResult:
        def __init__(self, text, company, score):
            self.score = score
            self.payload = {"text": text, "company": company}

    fake_results = []
    for i, obs in enumerate(observations):
        company_match = re.search(r"\[([^\]]+)\]:", obs["result"])
        company = company_match.group(1) if company_match else "Unknown"
        fake_results.append(ObsResult(obs["result"], company, float(i)))

    return generate_answer(query, fake_results, episodes=episodes)


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

    # ── Layer 4: ReAct loop ────────────────────────────────────
    agent_result = react_loop(resolved, episodes=episodes)

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
                "label":   f"{tc['tool']} (step {tc['step']})",
                "text":    tc["result"]
            })

    # ── Layer 5: Save results ──────────────────────────────────
    # Only save confident answers — never cache "I don't know"
    is_confident = DONT_KNOW_PHRASE not in answer.lower()
    if is_confident:
        # pipelines.embedder — module reference, always the real model
        qvec = pipelines.embedder.encode(query).tolist()
        rvec = pipelines.embedder.encode(resolved).tolist()
        save_to_cache_store(query, qvec, answer, pipelines.raw_cache)
        save_to_cache_store(resolved, rvec, answer, pipelines.resolved_cache)
        persist_cache_stores()
        save_episode_prod(resolved, answer, session_id)

    return {
        "pipeline":          "Agentic RAG",
        "query_type":        "agentic",
        "answer":            answer,
        "elapsed_sec":       round(time.perf_counter() - t0, 2),
        "cache_stage":       "miss",
        "is_confident":      is_confident,
        "episodes_used":     len(episodes),
        "companies":         companies_searched,
        "sources":           sources,
        "tool_calls":        tool_calls,
        "cycles":            cycles,
        "thoughts":          thoughts,
        "navigation_passes": None,
        "resolved_query":    resolved,
    }