"""Streamlit demo — UI only. Models/indexes/cache/memory live in the RAG worker."""
import json
import uuid
from pathlib import Path

import requests
import streamlit as st

ROOT = Path(__file__).parent
WORKER_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="EDGAR RAG Demo",
    page_icon="📄",
    layout="wide"
)

# ── Sidebar ───────────────────────────────────────────────────
st.sidebar.title("Settings")

compare_mode = st.sidebar.toggle(
    "Compare pipelines side by side", value=False
)

if not compare_mode:
    pipeline = st.sidebar.radio(
        "Pipeline",
        ["Adaptive RAG", "Vector RAG", "Vectorless RAG"],
        index=0,
    )
else:
    # In compare mode, choose which pipelines to compare
    compare_pipelines = st.sidebar.multiselect(
        "Compare these pipelines",
        ["Adaptive RAG", "Vector RAG", "Vectorless RAG"],
        default=["Vector RAG", "Vectorless RAG"],
    )

# Sample questions from eval set
with open(ROOT / "eval_set.json", encoding="utf-8") as f:
    eval_set = json.load(f)

st.sidebar.markdown("**Sample questions**")
for item in eval_set:
    label = f"Q{item['id']}: {item['question'][:45]}..."
    if st.sidebar.button(label, key=f"sample_{item['id']}"):
        st.session_state["pending_q"] = item["question"]

# Cache stats in sidebar
st.sidebar.markdown("---")
if st.sidebar.button("Show cache stats"):
    try:
        r = requests.get(f"{WORKER_URL}/cache_stats", timeout=5)
        if r.status_code == 200:
            stats = r.json()
            st.sidebar.metric("Hit rate", stats.get("hit_rate", "N/A"))
            st.sidebar.metric("Stage 1 hits", stats.get("stage1_hits", 0))
            st.sidebar.metric("Stage 2 hits", stats.get("stage2_hits", 0))
            st.sidebar.metric("Total queries", stats.get("total_queries", 0))
    except Exception:
        st.sidebar.warning("Could not fetch cache stats")


# ── Session helpers ───────────────────────────────────────────

def get_session_id() -> str:
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    return st.session_state.session_id


def chat_history_payload():
    out = []
    for m in st.session_state.get("messages", [])[-6:]:
        if m["role"] in ("user", "assistant") and isinstance(m.get("content"), str):
            out.append({"role": m["role"], "content": m["content"][:500]})
    return out


# ── Worker calls ──────────────────────────────────────────────

def call_ask(query: str, selected: str) -> dict:
    r = requests.post(
        f"{WORKER_URL}/ask",
        json={
            "query": query,
            "pipeline": selected,
            "session_id": get_session_id(),
            "chat_history": chat_history_payload(),
        },
        timeout=300,
    )
    r.raise_for_status()
    return r.json()


def call_compare(query: str, pipelines: list) -> dict:
    """
    Calls worker /compare endpoint with list of pipelines to compare.
    Returns dict keyed by pipeline name.
    """
    r = requests.post(
        f"{WORKER_URL}/compare",
        json={
            "query": query,
            "pipelines": pipelines,
            "session_id": get_session_id(),
            "chat_history": chat_history_payload(),
        },
        timeout=600,
    )
    r.raise_for_status()
    return r.json()


# ── Health check ──────────────────────────────────────────────

try:
    requests.get(f"{WORKER_URL}/health", timeout=2).raise_for_status()
except Exception:
    st.error(
        "Worker is not running.\n\n"
        "In another terminal:\n"
        "`cd c:\\RAGproject`\n"
        "`python -m uvicorn worker:app --host 127.0.0.1 --port 8000`"
    )
    st.stop()


# ── Header ────────────────────────────────────────────────────

st.title("EDGAR 2020 Risk Factors — RAG Demo")
st.caption(
    "Worker: Adaptive / Vector / Vectorless · "
    "disk Qdrant · semantic cache · episodic memory · "
    "8 companies · Item 1A"
)

if "messages" not in st.session_state:
    st.session_state.messages = []


# ── Display helpers ───────────────────────────────────────────

def show_sources(result: dict):
    """
    Shows pipeline metadata and retrieved sources for one result.
    All keys read with .get() so no KeyError if key is missing
    (e.g. cache hits don't have sources or resolved_query).
    """
    # ── Metadata row ──────────────────────────────────────────
    meta_parts = [
        f"**{result.get('pipeline', '?')}**",
        f"`{result.get('elapsed_sec', '?')}s`",
    ]

    # Query type (adaptive RAG only)
    if result.get("query_type"):
        type_emoji = {
            "simple_factual": "🔍",
            "comparative":    "⚖️",
            "structural":     "📋",
            "negative":       "❌",
        }.get(result["query_type"], "")
        meta_parts.append(
            f"route: `{result['query_type']}` {type_emoji}"
        )

    # Cache stage
    cache_stage = result.get("cache_stage")
    if cache_stage == "raw":
        meta_parts.append("cache: `raw hit` ⚡")
    elif cache_stage == "resolved":
        meta_parts.append("cache: `resolved hit` ⚡")
    elif cache_stage == "miss":
        meta_parts.append("cache: `miss`")

    # Episodes used from episodic memory
    episodes_used = result.get("episodes_used", 0)
    if episodes_used:
        meta_parts.append(f"episodes: `{episodes_used}`")

    # Navigation passes (vectorless paths only)
    nav_passes = result.get("navigation_passes")
    if nav_passes:
        meta_parts.append(f"passes: `{nav_passes}`")

    st.caption(" · ".join(meta_parts))

    # ── Confidence warning ────────────────────────────────────
    # Show warning if answer was low confidence
    # is_confident=False means DONT_KNOW_PHRASE was in the answer
    if result.get("is_confident") is False:
        st.warning(
            "⚠️ Low confidence answer — the system couldn't find "
            "enough information in the available filings."
        )

    # ── Resolved query (if different from original) ───────────
    # Show what the system actually searched for after pronoun resolution
    resolved = result.get("resolved_query", "")
    if resolved and resolved != result.get("original_query", resolved):
        st.info(f"🔄 Resolved query: *{resolved}*")

    # ── Cache hit — no sources to show ───────────────────────
    if cache_stage in ("raw", "resolved"):
        st.success(
            f"✅ Served from **{cache_stage} cache** "
            f"(full pipeline skipped)"
        )
        return

    # ── Companies retrieved ───────────────────────────────────
    companies = result.get("companies", [])
    if companies:
        st.markdown(f"**Retrieved companies:** `{companies}`")

    # ── Source chunks ─────────────────────────────────────────
    sources = result.get("sources") or []
    if not sources:
        st.caption("No sources retrieved.")
        return

    for i, src in enumerate(sources, 1):
        with st.expander(
            f"{i}. [{src['company']}] {src['label']}"
        ):
            text = src["text"]
            st.text(text[:1500] + ("..." if len(text) > 1500 else ""))


def show_result(result: dict, include_answer: bool = True):
    """
    Shows answer + sources for one result.
    include_answer=False when answer was already rendered
    as a chat message — avoids showing it twice.
    """
    if include_answer:
        st.markdown(result.get("answer", ""))
    st.markdown("#### Retrieved sources")
    show_sources(result)


# ── Replay chat history ───────────────────────────────────────

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(msg["content"])

        elif msg.get("compare"):
            # Compare mode — show results side by side
            results = msg.get("results", {})
            cols = st.columns(len(results))
            for col, (pipe_name, res) in zip(cols, results.items()):
                with col:
                    st.markdown(f"**{pipe_name}**")
                    show_result(res, include_answer=True)

        elif "result" in msg:
            # Single pipeline — answer already in msg["content"]
            st.markdown(msg["content"])
            show_result(msg["result"], include_answer=False)

        else:
            st.markdown(msg["content"])


# ── Chat input ────────────────────────────────────────────────

query = st.chat_input("Ask about a 2020 10-K risk factor...")
if "pending_q" in st.session_state:
    query = st.session_state.pop("pending_q")

if query:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving via worker..."):

            if compare_mode:
                # Compare selected pipelines side by side
                pipes = compare_pipelines if compare_pipelines else [
                    "Vector RAG", "Vectorless RAG"
                ]
                all_results = call_compare(query, pipes)

                cols = st.columns(len(pipes))
                for col, pipe_name in zip(cols, pipes):
                    res = all_results.get(pipe_name, {})
                    with col:
                        st.markdown(f"**{pipe_name}**")
                        show_result(res, include_answer=True)

                st.session_state.messages.append({
                    "role":    "assistant",
                    "content": "",
                    "compare": True,
                    "results": all_results,
                })

            else:
                # Single pipeline
                result = call_ask(query, pipeline)
                show_result(result, include_answer=True)
                st.session_state.messages.append({
                    "role":    "assistant",
                    "content": result.get("answer", ""),
                    "result":  result,
                })


# ── Eval results ──────────────────────────────────────────────

with st.expander("Eval results (9 questions)"):
    st.markdown(
        "| Pipeline | Score | Notes |\n"
        "|----------|-------|-------|\n"
        "| Vector RAG | **7/9** | Fails Q1, Q7 |\n"
        "| Vectorless RAG | **9/9** | All pass, Q8/Q9 need 2 passes |\n"
        "| Adaptive RAG | routes by query type | structural→vectorless, "
        "comparative→vector, negative→skip |"
    )