"""Streamlit demo — UI only. Models/indexes live in the RAG worker."""
import json
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


def call_ask(query: str, selected: str) -> dict:
    r = requests.post(
        f"{WORKER_URL}/ask",
        json={"query": query, "pipeline": selected},
        timeout=300,
    )
    r.raise_for_status()
    return r.json()


def call_compare(query: str) -> dict:
    r = requests.post(
        f"{WORKER_URL}/compare",
        json={"query": query},
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
        "`uvicorn worker:app --host 127.0.0.1 --port 8000`"
    )
    st.stop()

st.title("EDGAR 2020 Risk Factors — RAG Demo")
st.caption(
    "UI talks to local worker · Adaptive router picks Vector or Vectorless · "
    "8 companies · Item 1A only"
)

if "messages" not in st.session_state:
    st.session_state.messages = []


def show_sources(result):
    st.markdown(f"**Retrieved companies:** `{result['companies']}`")
    meta = f"{result['pipeline']} · {result['elapsed_sec']}s"
    if result.get("query_type"):
        meta += f" · route: `{result['query_type']}`"
    st.caption(meta)
    if result.get("navigation_passes"):
        st.caption(f"Navigation passes: {result['navigation_passes']}")
    for i, src in enumerate(result["sources"], 1):
        with st.expander(f"{i}. [{src['company']}] {src['label']}"):
            text = src["text"]
            st.text(text[:1500] + ("..." if len(text) > 1500 else ""))


def show_result(result):
    st.markdown(result["answer"])
    st.markdown("#### Retrieved sources")
    show_sources(result)


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            if msg.get("compare"):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Vector RAG**")
                    show_result(msg["vector"])
                with col2:
                    st.markdown("**Vectorless RAG**")
                    show_result(msg["vectorless"])
            elif "result" in msg:
                show_result(msg["result"])

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
                    show_result(rv)
                with col2:
                    st.markdown("**Vectorless RAG**")
                    show_result(rvl)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": rv["answer"][:300],
                    "compare": True,
                    "vector": rv,
                    "vectorless": rvl,
                })
            else:
                result = call_ask(query, pipeline)
                show_result(result)
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