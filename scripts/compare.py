"""Paired comparison of two per-question result files: bootstrap CI + sign test, Bonferroni over
--family. Copied from Personal-Knowledge-RAG's eval/harness/compare.py (same statistics and verdict).

A and B are <results.json>:<stage>, e.g.
  eval/agent_results.json:agent_first
  "<Personal-Knowledge-RAG>/rag-project/eval/runs/retrieval_baseline_current.json:reranked"

Usage: python scripts/compare.py A B <metric> [--metric-b M] [--split dev|test|all] [--family N]
"""

import argparse
import json
import random
from math import comb
from pathlib import Path


def load(spec: str) -> dict[str, dict]:
    path, _, stage = spec.rpartition(":")  # rpartition: Windows paths contain "D:"
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    return {r["id"]: {"split": r["split"], **r["metrics"][stage]} for r in rows}


def sign_test(wins: int, losses: int) -> float:
    n = wins + losses
    tail = sum(comb(n, k) for k in range(min(wins, losses) + 1)) / 2**n if n else 1.0
    return min(1.0, 2 * tail)


def bootstrap_ci(diffs: list[float], alpha: float, reps: int = 10_000, seed: int = 0) -> tuple[float, float]:
    rng = random.Random(seed)  # nosec B311
    means = sorted(sum(rng.choices(diffs, k=len(diffs))) / len(diffs) for _ in range(reps))
    return means[int(alpha / 2 * reps)], means[int((1 - alpha / 2) * reps) - 1]


def verdict(diffs: list[float], p: float, low: float, high: float, alpha: float, min_effect: float) -> str:
    mean = sum(diffs) / len(diffs)
    if p >= alpha or low <= 0 <= high:
        return "no evidence of a difference at this sample size"
    if abs(mean) < min_effect:
        return f"significant but smaller than the minimum useful effect ({min_effect})"
    return "A higher" if mean > 0 else "B higher"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("metric")
    p.add_argument("--metric-b", help="B's metric if it differs (e.g. recall@5 vs recall@3)")
    p.add_argument("--split", choices=["dev", "test", "all"], default="all")
    p.add_argument("--family", type=int, default=1, help="number of comparisons in this family (Bonferroni)")
    p.add_argument("--min-effect", type=float, default=0.0, help="smallest difference worth acting on")
    args = p.parse_args()

    a, b = load(args.a), load(args.b)
    mb = args.metric_b or args.metric
    sa = {q for q in a if args.split in ("all", a[q]["split"])}
    sb = {q for q in b if args.split in ("all", b[q]["split"])}
    ids = sorted((q for q in sa & sb if args.metric in a[q] and mb in b[q]), key=lambda q: int(q[1:]))
    if len(ids) < 2:
        raise SystemExit(f"only {len(ids)} paired questions with metric {args.metric!r}; nothing to compare")

    diffs = [a[q][args.metric] - b[q][mb] for q in ids]
    alpha = 0.05 / args.family
    wins, losses = sum(d > 0 for d in diffs), sum(d < 0 for d in diffs)
    pval = sign_test(wins, losses)
    low, high = bootstrap_ci(diffs, alpha)

    print(f"A={args.a}  B={args.b}  metric={args.metric}{'/' + mb if mb != args.metric else ''}  split={args.split}")
    print(f"  pairs={len(ids)}  in both runs={len(sa & sb)}  only A={len(sa - sb)}  only B={len(sb - sa)}")
    print(f"  mean A={sum(a[q][args.metric] for q in ids) / len(ids):.3f}  B={sum(b[q][mb] for q in ids) / len(ids):.3f}")
    print(f"  diff A-B = {sum(diffs) / len(diffs):+.3f}   {100 * (1 - alpha):.1f}% CI [{low:+.3f}, {high:+.3f}]")
    print(f"  A higher on {wins}, B higher on {losses}, tied {len(ids) - wins - losses}; sign test p={pval:.4f} (alpha {alpha:.4f})")
    print(f"  verdict: {verdict(diffs, pval, low, high, alpha, args.min_effect)}")
    for q, d in sorted(zip(ids, diffs), key=lambda t: -abs(t[1]))[:20]:
        if d:
            print(f"    {q}: A={a[q][args.metric]:.2f}  B={b[q][mb]:.2f}")


def _selfcheck() -> None:
    assert abs(sign_test(8, 1) - 0.0390625) < 1e-9 and sign_test(0, 0) == 1.0  # nosec B101
    assert bootstrap_ci([0.5] * 10, 0.05) == (0.5, 0.5)  # nosec B101
    assert verdict([0.0] * 10, 1.0, 0.0, 0.0, 0.05, 0.0).startswith("no evidence")  # nosec B101
    assert verdict([1.0] * 10, sign_test(10, 0), 1.0, 1.0, 0.05, 0.0) == "A higher"  # nosec B101
    assert "smaller" in verdict([0.1] * 10, sign_test(10, 0), 0.1, 0.1, 0.05, 0.5)  # nosec B101


if __name__ == "__main__":
    _selfcheck()
    main()
