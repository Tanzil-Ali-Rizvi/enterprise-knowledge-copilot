"""Evaluation harness: retrieval quality, answer correctness, citation accuracy, abstention and
access-control leakage, across retrieval configurations.

    python -m kcopilot.evaluation                 # compare configs, write evals/results/
    python -m kcopilot.evaluation --gate          # exit 1 if quality gates fail (used in CI)
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

from .config import Settings, get_settings
from .pipeline import Copilot

ANSWERABLE = {"factual", "paraphrase", "acl_allowed"}

CONFIGS = {
    "bm25": {"retrieval_mode": "bm25", "reranker": "none"},
    "vector": {"retrieval_mode": "vector", "reranker": "none"},
    "hybrid": {"retrieval_mode": "hybrid", "reranker": "none"},
    "hybrid+rerank": {"retrieval_mode": "hybrid"},  # uses the configured reranker
}

# Quality gates for CI (applied to the "hybrid+rerank" config).
GATES = {
    "hit@5": 0.90,
    "answer_accuracy": 0.80,
    "citation_accuracy": 0.80,
    "acl_leak_rate": 0.0,
    "abstention_accuracy": 0.80,
}


def load_cases(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def evaluate(copilot: Copilot, cases: list[dict]) -> dict:
    rows, latencies = [], []
    for case in cases:
        ans = copilot.ask(case["question"], case["user"])
        latencies.append(ans.timings_ms["total_ms"])
        sources = [copilot.index.chunks[cid].source for cid in ans.retrieved]
        cited = [c.source for c in ans.citations]
        text = ans.answer.lower()
        row = {
            "id": case["id"],
            "category": case["category"],
            "abstained": ans.abstained,
            "confidence": ans.confidence,
            "answer": ans.answer,
            "cited": cited,
        }

        if case["category"] in ANSWERABLE:
            exp = case["expected_source"]
            rank = next((i + 1 for i, s in enumerate(sources) if s == exp), None)
            row.update(
                rank=rank,
                answer_ok=not ans.abstained and all(k.lower() in text for k in case["expected_keywords"]),
                citation_ok=exp in cited,
            )
        forbidden = set(case.get("forbidden_sources", []))
        if forbidden or case.get("forbidden_keywords"):
            leaked = (
                bool(forbidden & set(sources))
                or bool(forbidden & set(cited))
                or any(k.lower() in text for k in case.get("forbidden_keywords", []))
            )
            row["leak"] = leaked
        if "should_abstain" in case:
            row["abstain_ok"] = ans.abstained == case["should_abstain"]
        rows.append(row)

    ans_rows = [r for r in rows if r["category"] in ANSWERABLE]
    leak_rows = [r for r in rows if "leak" in r]
    abst_rows = [r for r in rows if "abstain_ok" in r]

    def frac(xs) -> float:
        xs = list(xs)
        return round(sum(xs) / len(xs), 3) if xs else 0.0

    k = copilot.settings.top_k
    metrics = {
        "cases": len(rows),
        "hit@1": frac(r["rank"] == 1 for r in ans_rows),
        f"hit@{k}": frac(r["rank"] is not None for r in ans_rows),
        "mrr": frac((1 / r["rank"]) if r["rank"] else 0 for r in ans_rows),
        "answer_accuracy": frac(r["answer_ok"] for r in ans_rows),
        "citation_accuracy": frac(r["citation_ok"] for r in ans_rows),
        "abstention_accuracy": frac(r["abstain_ok"] for r in abst_rows),
        "acl_leak_rate": frac(r["leak"] for r in leak_rows),
        "latency_p50_ms": round(statistics.median(latencies), 1),
        "latency_p95_ms": round(sorted(latencies)[max(0, int(len(latencies) * 0.95) - 1)], 1),
    }
    if "hit@5" not in metrics:
        metrics["hit@5"] = metrics[f"hit@{k}"]
    return {"metrics": metrics, "rows": rows}


def run(configs: list[str], golden: Path, out_dir: Path, base: Settings | None = None) -> dict[str, dict]:
    base = base or get_settings()
    cases = load_cases(golden)
    results = {}
    for name in configs:
        settings = base.model_copy(update=CONFIGS[name])
        cp = Copilot(settings=settings, persist=False)
        results[name] = evaluate(cp, cases)
        results[name]["settings"] = {
            "embedder": cp.embedder.name,
            "reranker": cp.reranker.name,
            "llm": cp.llm.name,
            "mode": settings.retrieval_mode,
        }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (out_dir / "report.md").write_text(render_report(results), encoding="utf-8")
    return results


def render_report(results: dict[str, dict]) -> str:
    cols = [
        "hit@1",
        "hit@5",
        "mrr",
        "answer_accuracy",
        "citation_accuracy",
        "abstention_accuracy",
        "acl_leak_rate",
        "latency_p50_ms",
        "latency_p95_ms",
    ]
    any_r = next(iter(results.values()))
    lines = [
        "# Evaluation report",
        "",
        f"Golden set: {any_r['metrics']['cases']} cases (factual, paraphrase, access-control allowed/denied, "
        "prompt injection, out-of-scope).  ",
        f"Embedder: `{any_r['settings']['embedder']}` · LLM: `{any_r['settings']['llm']}`",
        "",
        "| Config | " + " | ".join(cols) + " |",
        "|---|" + "---|" * len(cols),
    ]
    for name, r in results.items():
        m = r["metrics"]
        lines.append(f"| {name} | " + " | ".join(str(m[c]) for c in cols) + " |")
    lines += ["", "## Failures (best config)", ""]
    best = results.get("hybrid+rerank") or any_r
    fails = [r for r in best["rows"] if r.get("answer_ok") is False or r.get("leak") or r.get("abstain_ok") is False]
    if not fails:
        lines.append("None.")
    for r in fails:
        lines.append(f"- `{r['id']}` ({r['category']}): {r['answer'][:160]}")
    return "\n".join(lines) + "\n"


def check_gates(metrics: dict) -> list[str]:
    failures = []
    for key, threshold in GATES.items():
        value = metrics[key]
        ok = value <= threshold if key == "acl_leak_rate" else value >= threshold
        if not ok:
            failures.append(f"{key}={value} (gate {'<=' if key == 'acl_leak_rate' else '>='} {threshold})")
    return failures


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run the RAG evaluation suite")
    p.add_argument("--golden", type=Path, default=Path("evals/golden.jsonl"))
    p.add_argument("--out", type=Path, default=Path("evals/results"))
    p.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
    p.add_argument("--gate", action="store_true", help="fail (exit 1) if quality gates are not met")
    args = p.parse_args(argv)

    results = run(args.configs, args.golden, args.out)
    print(render_report(results))
    if args.gate:
        target = results.get("hybrid+rerank") or next(iter(results.values()))
        failures = check_gates(target["metrics"])
        if failures:
            print("QUALITY GATE FAILED:\n  " + "\n  ".join(failures))
            return 1
        print("Quality gates passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
