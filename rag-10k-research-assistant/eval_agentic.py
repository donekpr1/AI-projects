"""
Light Agentic RAG evaluation against eval_set.json.

Runs each gold question through the worker /agentic endpoint,
grades vs expected_answer (LLM judge + negative heuristics),
logs cycles / latency / tools, writes agentic_eval_results.json.

Usage (worker must be running):
  python eval_agentic.py
  python eval_agentic.py --compare-adaptive
  python eval_agentic.py --ids 3,6,8

Tip: clear cache_store.json before a clean A/B run so cache hits
do not inflate Agentic scores or hide tool behavior.
"""

from __future__ import annotations

import argparse
import json
import time
import uuid
from pathlib import Path

import requests
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

ROOT = Path(__file__).parent
EVAL_SET = ROOT / "eval_set.json"
OUT_PATH = ROOT / "agentic_eval_results.json"

DONT_KNOW_MARKERS = (
    "i don't have enough information",
    "not available",
    "not enough information",
    "empty in the dataset",
    "out of corpus",
    "no information",
)

llm = OpenAI()


def _call_pipeline(
    query: str, pipeline: str, session_id: str, worker_url: str
) -> dict:
    if pipeline == "Agentic RAG":
        url = f"{worker_url}/agentic"
        payload = {
            "query": query,
            "session_id": session_id,
            "chat_history": None,
        }
    else:
        url = f"{worker_url}/ask"
        payload = {
            "query": query,
            "pipeline": pipeline,
            "session_id": session_id,
            "chat_history": None,
        }
    r = requests.post(url, json=payload, timeout=600)
    r.raise_for_status()
    return r.json()


def _looks_like_refusal(text: str) -> bool:
    t = (text or "").lower()
    return any(m in t for m in DONT_KNOW_MARKERS)


def grade_answer(
    question: str,
    expected: str,
    generated: str,
    qtype: str,
) -> dict:
    """
    Returns {passed: bool, method: str, reason: str}.
    Negatives: pass if answer is a refusal / empty-section style.
    Others: LLM PASS/FAIL vs expected (key facts, not wording match).
    """
    gen = (generated or "").strip()
    exp = (expected or "").strip()

    if qtype == "negative":
        passed = _looks_like_refusal(gen)
        return {
            "passed": passed,
            "method": "heuristic_negative",
            "reason": (
                "Refusal / empty-scope language present"
                if passed
                else "Expected a don't-know / not-available style answer"
            ),
        }

    if not gen:
        return {
            "passed": False,
            "method": "empty",
            "reason": "Empty generated answer",
        }

    prompt = f"""You grade a RAG system answer against a gold expected answer.
Return ONLY JSON: {{"passed": true/false, "reason": "one short sentence"}}

Rules:
- passed=true if the generated answer captures the key facts in the expected answer
  (wording may differ; synonyms OK).
- passed=false if key facts are missing, wrong, or the answer refuses when it should know.
- For comparative questions, both sides of the contrast must be substantially correct.

Question: {question}

Expected: {exp}

Generated: {gen}
"""
    try:
        resp = llm.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=120,
        )
        raw = (resp.choices[0].message.content or "").strip()
        # Tolerate markdown fences
        if raw.startswith("```"):
            raw = raw.strip("`")
            if raw.startswith("json"):
                raw = raw[4:].strip()
        data = json.loads(raw)
        return {
            "passed": bool(data.get("passed")),
            "method": "llm_judge",
            "reason": str(data.get("reason", ""))[:240],
        }
    except Exception as e:
        # Fallback: soft keyword overlap on distinctive tokens
        exp_l = exp.lower()
        gen_l = gen.lower()
        keys = [
            w for w in (
                "panasonic", "80%", "over 80", "item 7", "item 8",
                "gaming", "automotive", "summary risk", "competition",
                "manufacturing", "financial",
            )
            if w in exp_l
        ]
        if keys:
            hits = sum(1 for k in keys if k in gen_l)
            passed = hits >= max(1, len(keys) // 2)
        else:
            passed = False
        return {
            "passed": passed,
            "method": "keyword_fallback",
            "reason": f"LLM judge failed ({e}); keyword heuristic used",
        }


def eval_one(item: dict, pipeline: str, worker_url: str) -> dict:
    session_id = f"eval-{pipeline[:3].lower()}-{item['id']}-{uuid.uuid4().hex[:8]}"
    t0 = time.perf_counter()
    result = _call_pipeline(item["question"], pipeline, session_id, worker_url)
    wall = round(time.perf_counter() - t0, 2)

    answer = result.get("answer") or ""
    grade = grade_answer(
        item["question"],
        item["expected_answer"],
        answer,
        item.get("type", "factual"),
    )

    tools = result.get("tool_calls") or []
    tool_names = [t.get("tool") for t in tools if t.get("tool")]

    return {
        "id": item["id"],
        "type": item.get("type"),
        "pipeline": pipeline,
        "question": item["question"],
        "expected_answer": item["expected_answer"],
        "generated_answer": answer,
        "passed": grade["passed"],
        "grade_method": grade["method"],
        "grade_reason": grade["reason"],
        "elapsed_sec": result.get("elapsed_sec", wall),
        "wall_sec": wall,
        "cycles": result.get("cycles", 0),
        "cache_stage": result.get("cache_stage"),
        "tools": tool_names,
        "companies": result.get("companies") or [],
        "is_confident": result.get("is_confident"),
    }


def main():
    parser = argparse.ArgumentParser(description="Eval Agentic RAG on gold set")
    parser.add_argument(
        "--compare-adaptive",
        action="store_true",
        help="Also run Adaptive RAG for side-by-side scoring",
    )
    parser.add_argument(
        "--ids",
        type=str,
        default="",
        help="Comma-separated question ids (default: all)",
    )
    parser.add_argument(
        "--worker",
        type=str,
        default="http://127.0.0.1:8000",
        help="Worker base URL",
    )
    args = parser.parse_args()
    worker_url = args.worker.rstrip("/")

    # Health check
    try:
        h = requests.get(f"{worker_url}/health", timeout=5)
        h.raise_for_status()
    except Exception as e:
        raise SystemExit(
            f"Worker not reachable at {worker_url}. "
            f"Start: python -m uvicorn worker:app --host 127.0.0.1 --port 8000\n{e}"
        )

    items = json.loads(EVAL_SET.read_text(encoding="utf-8"))
    if args.ids.strip():
        want = {int(x) for x in args.ids.split(",") if x.strip()}
        items = [i for i in items if i["id"] in want]

    pipelines = ["Agentic RAG"]
    if args.compare_adaptive:
        pipelines.append("Adaptive RAG")

    all_rows = []
    for pipe in pipelines:
        print(f"\n=== {pipe} ({len(items)} questions) ===")
        for item in items:
            print(f"  Q{item['id']} ...", end=" ", flush=True)
            row = eval_one(item, pipe, worker_url)
            all_rows.append(row)
            mark = "PASS" if row["passed"] else "FAIL"
            print(
                f"{mark} | {row['elapsed_sec']}s | "
                f"cycles={row['cycles']} | cache={row['cache_stage']} | "
                f"{row['grade_reason'][:80]}"
            )

    # Summaries
    summaries = {}
    for pipe in pipelines:
        rows = [r for r in all_rows if r["pipeline"] == pipe]
        n = len(rows) or 1
        passed = sum(1 for r in rows if r["passed"])
        summaries[pipe] = {
            "score": f"{passed}/{len(rows)}",
            "accuracy": round(passed / n, 3),
            "avg_elapsed_sec": round(
                sum(r["elapsed_sec"] for r in rows) / n, 2
            ),
            "avg_cycles": round(
                sum(r.get("cycles") or 0 for r in rows) / n, 2
            ),
            "cache_hits": sum(
                1 for r in rows if r.get("cache_stage") in ("raw", "resolved")
            ),
        }

    out = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "worker": worker_url,
        "summaries": summaries,
        "results": all_rows,
        "notes": (
            "LLM-judge for factual/comparative; heuristic refusal check "
            "for negative questions. Clear cache_store.json for a cold run."
        ),
    }
    OUT_PATH.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("\n=== Summary ===")
    for pipe, s in summaries.items():
        print(
            f"  {pipe}: {s['score']} "
            f"(acc={s['accuracy']}, avg {s['avg_elapsed_sec']}s, "
            f"avg cycles={s['avg_cycles']}, cache_hits={s['cache_hits']})"
        )
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
