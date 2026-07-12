"""Streamlit demo — UI only. Models/indexes/cache/memory live in the RAG worker."""
import json
import uuid
from pathlib import Path

import requests
import streamlit as st

ROOT = Path(__file__).parent
WORKER_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="EDGAR RAG Demo", page_icon="📄", layout="wide")

st.sidebar.title("Settings")
compare_mode = st.sidebar.toggle("Compare Vector vs Vectorless", value=False)
if not compare_mode:
    pipeline = st.sidebar.radio(
        "Pipeline",
        ["Adaptive RAG", "Vector RAG", "Vectorless RAG"],
        index=0,
    )

with open(ROOT / "eval_set.json", encoding="utf-8") as f:
    eval_set = json.load(f)

st.sidebar.markdown("**Sample questions**")
for item in eval_set:
    label = f"Q{item['id']}: {item['question'][:45]}..."
    if st.sidebar.button(label, key=f"sample_{item['id']}"):
        st.session_state["pending_q"] = item["question"]


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


def call_compare(query: str) -> dict:
    r = requests.post(
        f"{WORKER_URL}/compare",
        json={
            "query": query,
            "session_id": get_session_id(),
            "chat_history": chat_history_payload(),
        },
        timeout=600,
    )
    r.raise_for_status()
    return r.json()


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

st.title("EDGAR 2020 Risk Factors — RAG Demo")
st.caption(
    "Worker: Adaptive / Vector / Vectorless · disk Qdrant · "
    "semantic cache · episodic memory · 8 companies · Item 1A"
)

if "messages" not in st.session_state:
    st.session_state.messages = []


def show_sources(result):
    st.markdown(f"**Retrieved companies:** `{result.get('companies', [])}`")
    meta = f"{result.get('pipeline', '?')} · {result.get('elapsed_sec', '?')}s"
    if result.get("query_type"):
        meta += f" · route: `{result['query_type']}`"
    if result.get("cache_stage"):
        meta += f" · cache: `{result['cache_stage']}`"
    if result.get("episodes_used"):
        meta += f" · episodes: `{result['episodes_used']}`"
    st.caption(meta)

    if result.get("navigation_passes"):
        st.caption(f"Navigation passes: {result['navigation_passes']}")

    if result.get("cache_stage") in ("raw", "resolved"):
        st.info(f"Served from **{result['cache_stage']} cache** (full pipeline skipped)")
        return

    for i, src in enumerate(result.get("sources") or [], 1):
        with st.expander(f"{i}. [{src['company']}] {src['label']}"):
            text = src["text"]
            st.text(text[:1500] + ("..." if len(text) > 1500 else ""))


def show_result(result, include_answer: bool = True):
    """include_answer=False when answer was already shown as msg['content']."""
    if include_answer:
        st.markdown(result["answer"])
    st.markdown("#### Retrieved sources")
    show_sources(result)


# Replay chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(msg["content"])
        elif msg.get("compare"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Vector RAG**")
                show_result(msg["vector"], include_answer=True)
            with col2:
                st.markdown("**Vectorless RAG**")
                show_result(msg["vectorless"], include_answer=True)
        elif "result" in msg:
            st.markdown(msg["content"])  # answer once
            show_result(msg["result"], include_answer=False)  # sources only
        else:
            st.markdown(msg["content"])

query = st.chat_input("Ask about a 2020 10-K risk factor...")
if "pending_q" in st.session_state:
    query = st.session_state.pop("pending_q")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)
    with st.chat_message("assistant"):
        with st.spinner("Retrieving via worker..."):
            if compare_mode:
                both = call_compare(query)
                rv = both["vector"]
                rvl = both["vectorless"]
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Vector RAG**")
                    show_result(rv, include_answer=True)
                with col2:
                    st.markdown("**Vectorless RAG**")
                    show_result(rvl, include_answer=True)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "",  # answers live in vector/vectorless
                    "compare": True,
                    "vector": rv,
                    "vectorless": rvl,
                })
            else:
                result = call_ask(query, pipeline)
                show_result(result, include_answer=True)  # answer once here
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "result": result,
                })

with st.expander("Eval results (9 questions)"):
    st.markdown(
        "| Pipeline | Score |\n"
        "|----------|-------|\n"
        "| Vector RAG | **7/9** |\n"
        "| Vectorless RAG | **9/9** |\n"
        "| Adaptive RAG | routes by query type |"
    )