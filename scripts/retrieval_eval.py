"""Retrieval eval: recall@k, success@k and MRR on every answerable question.

The dataset is the Personal-Knowledge-RAG golden dataset (eval/data/golden_dataset.json): each
answerable item labels evidence spans as {"source": file name, "evidence": text}. A retrieved chunk
is a hit when it comes from that file and contains the span (case, punctuation and whitespace
ignored), so the labels do not depend on how a pipeline chunks. The metric definitions copy that
project's harness so the numbers are comparable; they are provisional until a shared eval kit
re-measures every RAG variant.

Two runs:
  default  ($0, no LLM call): the agent's retrieve tool on the raw question, every stage scored
           (dense, bm25, fused, reranked); the tool's LLM rewrite is skipped.
           -> eval/retrieval_results.json
  --agent  (LLM calls): the LangGraph ReAct agent (src/agent/react_agent.py) answers each question
           and every retrieve call it makes is captured. Scored: agent_first (the first call's
           reranked top 5) and agent_all (the chunks of all calls; recall@all = any chunk the agent
           saw). Also recorded: tool calls, grader verdicts, LLM calls (agent / rewrite / grade) and
           tokens, latency, outcome, answer.
           -> eval/agent_results.json, saved after every question; a re-run skips saved ones.
           A provider error (e.g. 429) discards that question, waits 60 s and retries once;
           a second error stops the run (re-run later to resume).

Usage: python scripts/retrieval_eval.py <golden_dataset.json> [--agent [--limit N]] [--out FILE]
"""

import argparse
import hashlib
import json
import os
import re
import statistics
import sys
import time
import unicodedata
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def norm(text: str) -> str:
    return re.sub(r"[\W_]+", "", unicodedata.normalize("NFKC", text).casefold())


def chunk_source(chunk: dict) -> str:
    meta = chunk["metadata"]
    return re.split(r"[\\/]", meta.get("filename") or meta.get("source", ""))[-1]


def score(item: dict, ranked: list[dict], ks: tuple[int, ...]) -> dict[str, float]:
    cols = [(chunk_source(c), norm(c["text"])) for c in ranked]
    matrix = [
        [src == rel["source"] and norm(rel["evidence"]) in text for src, text in cols]
        for rel in item["relevant"]
    ]
    any_of = item.get("any_of", False)
    out: dict[str, float] = {}
    for k in ks:
        covered = [any(row[:k]) for row in matrix]
        out[f"recall@{k}"] = float(any(covered)) if any_of else sum(covered) / len(covered)
        out[f"success@{k}"] = float(any(covered) if any_of else all(covered))
    first = [row.index(True) + 1 for row in matrix if True in row]
    out["mrr"] = 1 / min(first) if first else 0.0
    return out


# deeper k per stage, as in Personal-Knowledge-RAG's harness; recall@1/3/5 everywhere
STAGE_KS = {"dense": (20,), "bm25": (20,), "fused": (10,), "reranked": ()}


def stage_ks(stage: str, depth: int) -> tuple[int, ...]:
    return tuple(sorted({1, 3, 5} | {k for k in STAGE_KS.get(stage, ()) if k <= depth}))


def agent_metrics(item: dict, seen: list[tuple[str, list[dict]]]) -> dict[str, dict[str, float]]:
    union, texts = [], set()
    for _, ranked in seen:  # call order, a chunk counted once
        for c in ranked:
            if c["text"] not in texts:
                texts.add(c["text"])
                union.append(c)
    n = max(len(union), 1)
    s = score(item, union, (n,))
    return {
        "agent_first": score(item, seen[0][1] if seen else [], (1, 3, 5)),
        "agent_all": {"recall@all": s[f"recall@{n}"], "success@all": s[f"success@{n}"], "chunks": float(len(union))},
    }


def summary(rows: list[dict]) -> None:
    for stage, metrics in rows[0]["metrics"].items():
        cells = "  ".join(f"{m}={sum(r['metrics'][stage][m] for r in rows) / len(rows):.2f}" for m in metrics)
        print(f"  {stage:<11} {cells}")


def run_agent_eval(items: list[dict], ids: dict, settings: dict, out: Path, limit: int | None) -> None:
    import requests

    import src.agent.tools as tools
    from config import REWRITE_SYSTEM
    from src.agent.prompts import GRADE_SYSTEM
    from src.agent.react_agent import run_agent
    from src.embedding.embedder import embed_query
    from src.llm import client

    rows = json.loads(out.read_text(encoding="utf-8")) if out.exists() else []
    if rows and (rows[0]["dataset"], rows[0]["corpus"]) != (ids["dataset"], ids["corpus"]):
        raise SystemExit(f"{out} is for another dataset or corpus; move it away or pass --out")
    done = {r["id"] for r in rows}

    calls: list[dict] = []
    seen: list[tuple[str, list[dict]]] = []
    real_post, real_stages = requests.post, tools.retrieve_stages

    def post(url, *a, **kw):  # records every pipeline LLM call (status, tokens)
        if url != client.ACTIVE_URL:
            return real_post(url, *a, **kw)
        t = time.perf_counter()
        r = real_post(url, *a, **kw)
        usage = r.json().get("usage", {}) if r.ok else {}
        system = kw["json"]["messages"][0]["content"]
        calls.append({
            "role": {REWRITE_SYSTEM: "rewrite", GRADE_SYSTEM: "grade"}.get(system, "agent"),
            "status": r.status_code,
            "ms": round((time.perf_counter() - t) * 1000),
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
        })
        return r

    def stages(query, *a, **kw):  # captures what each retrieve call returned
        result = real_stages(query, *a, **kw)
        seen.append((query, result["reranked"]))
        return result

    requests.post, tools.retrieve_stages = post, stages
    embed_query("warm up")  # model load stays out of the first question's latency

    todo = [i for i in items if i["id"] not in done][:limit]
    for n, item in enumerate(todo, 1):
        for attempt in (1, 2):
            calls.clear()
            seen.clear()
            t = time.perf_counter()
            try:
                result = run_agent(item["question"])
                errors = [c["status"] for c in calls if c["status"] != 200]
            except requests.RequestException as e:
                result, errors = None, [str(e)[:200]]
            latency_ms = round((time.perf_counter() - t) * 1000)
            if not errors:
                break
            print(f"{item['id']}: provider error {errors}; " + ("waiting 60 s, retrying" if attempt == 1 else "stopping"))
            if attempt == 1:
                time.sleep(60)
        else:
            raise SystemExit(f"stopped: {len(rows)} questions saved in {out}; re-run later to resume")

        steps = [s["parsed"] for s in result["trajectory"]]
        rows.append({
            "id": item["id"],
            "split": item["split"],
            "query": item["question"],
            **ids,
            "settings": settings,
            "metrics": agent_metrics(item, seen),
            "agent": {
                # answer | action (hit MAX_ITERATIONS) | unknown (unparseable output)
                "outcome": steps[-1]["type"],
                "iterations": result["iterations"],
                "tool_calls": [{"tool": s["action"], "input": s["action_input"]} for s in steps if s["type"] == "action"],
                "retrieve_queries": [q for q, _ in seen],  # after the tool's LLM rewrite
                "grades": [s["grade"] for s in result["trajectory"] if "grade" in s],
                "answer": result["answer"],
            },
            "llm": {
                "calls": len(calls),
                "agent_calls": sum(c["role"] == "agent" for c in calls),
                "rewrite_calls": sum(c["role"] == "rewrite" for c in calls),
                "grade_calls": sum(c["role"] == "grade" for c in calls),
                "prompt_tokens": sum(c["prompt_tokens"] for c in calls),
                "completion_tokens": sum(c["completion_tokens"] for c in calls),
                "llm_ms": sum(c["ms"] for c in calls),
            },
            "latency_ms": latency_ms,
        })
        out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        llm = rows[-1]["llm"]
        print(f"[{n}/{len(todo)}] {item['id']} calls={llm['calls']} tokens={llm['prompt_tokens'] + llm['completion_tokens']} {latency_ms} ms")

    print(f"{out.name}: {ids} {settings}  n={len(rows)}")
    summary(rows)
    for key in ("calls", "agent_calls", "rewrite_calls", "grade_calls", "prompt_tokens", "completion_tokens"):
        print(f"  {key}/question={sum(r['llm'].get(key, 0) for r in rows) / len(rows):.1f}")
    print(f"  latency_ms median={statistics.median(r['latency_ms'] for r in rows):.0f}")
    outcomes = [r["agent"]["outcome"] for r in rows]
    print(f"  outcomes: { {o: outcomes.count(o) for o in set(outcomes)} }")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("dataset", type=Path)
    p.add_argument("--agent", action="store_true", help="run the LangGraph agent (LLM calls)")
    p.add_argument("--limit", type=int, help="--agent: at most N new questions (a pilot)")
    p.add_argument("--out", type=Path)
    args = p.parse_args()

    os.environ["HF_HUB_OFFLINE"] = "1"  # models come from the local cache
    from config import CHUNK_OVERLAP, CHUNK_SIZE, CHUNKS_PATH, TOP_K

    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    texts = sorted(c["text"] for c in json.loads(CHUNKS_PATH.read_text(encoding="utf-8")))
    ids = {
        # same id as Personal-Knowledge-RAG's store.content_id (parsed, so CRLF-independent)
        "dataset": hashlib.sha256(json.dumps(dataset, sort_keys=True).encode()).hexdigest()[:10],
        "corpus": f"{len(texts)}:{hashlib.sha256(chr(0).join(texts).encode()).hexdigest()[:10]}",
    }
    items = [i for i in dataset if i["expected_outcome"] == "answer"]

    if args.agent:
        from src.agent.react_agent import MAX_ITERATIONS
        from src.llm.client import ACTIVE_MODEL, PIPELINE_PROVIDER

        settings = {"chunk": f"{CHUNK_SIZE}/{CHUNK_OVERLAP}", "top_k": TOP_K, "agent": "langgraph-react-grade",
                    "llm": f"{PIPELINE_PROVIDER}/{ACTIVE_MODEL}", "max_iterations": MAX_ITERATIONS}
        out = args.out or ROOT / "eval" / "agent_results.json"
        return run_agent_eval(items, ids, settings, out, args.limit)

    from src.pipeline.advanced_rag import retrieve_stages

    settings = {"chunk": f"{CHUNK_SIZE}/{CHUNK_OVERLAP}", "top_k": TOP_K, "rewrite": False}
    rows = []
    for item in items:
        stages = retrieve_stages(item["question"], top_k=TOP_K)
        rows.append({
            "id": item["id"],
            "split": item["split"],
            "query": item["question"],
            **ids,
            "settings": settings,
            "metrics": {s: score(item, r, stage_ks(s, len(r))) for s, r in stages.items()},
        })

    out = args.out or ROOT / "eval" / "retrieval_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"{out.name}: {ids} {settings}  n={len(rows)}")
    summary(rows)


if __name__ == "__main__":
    main()
