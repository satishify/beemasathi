"""Compare prompt strategies on the same retrieved context."""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval.judge import _call_judge, _rule_score
from rag.generator import PROMPT_STRATEGIES, BimaSaathiGenerator
from rag.retriever import PolicyRetriever

GOLD_PATH = ROOT / "eval" / "gold_cases.json"
REPORT_DIR = ROOT / "eval" / "reports"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare prompt strategies.")
    parser.add_argument(
        "--strategies",
        default="baseline,zero_shot,structured",
        help="Comma-separated strategy names.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Only the first N cases.")
    parser.add_argument("--judge", action="store_true", help="Also run the LLM judge.")
    parser.add_argument("--k", type=int, default=8)
    args = parser.parse_args()

    strategies = [name.strip() for name in args.strategies.split(",") if name.strip()]
    unknown = [name for name in strategies if name not in PROMPT_STRATEGIES]
    if unknown:
        raise SystemExit(f"Unknown strategies: {unknown}. Known: {list(PROMPT_STRATEGIES)}")

    cases = json.loads(GOLD_PATH.read_text(encoding="utf-8"))["cases"]
    if args.limit:
        cases = cases[: args.limit]

    retriever = PolicyRetriever()
    generator = BimaSaathiGenerator()
    catalog = retriever.store.get_catalog()

    # Retrieve once so every strategy sees identical evidence.
    cached = []
    for case in cases:
        chunks = retriever.retrieve(
            query=case["question"], insurer=case.get("insurer"), n_results=args.k
        )
        cached.append((case, chunks))
    print(f"Retrieved context for {len(cached)} cases.\n")

    results = {}
    for strategy in strategies:
        rows = []
        started = time.perf_counter()
        for case, chunks in cached:
            step = time.perf_counter()
            result = generator.generate(
                chunks, case["question"], catalog=catalog, strategy=strategy
            )
            latency_ms = round((time.perf_counter() - step) * 1000)
            rules = _rule_score(case, result)
            judge = _call_judge(generator, case, result, chunks) if args.judge else {}
            passed = rules["pass"] and (not args.judge or judge.get("verdict") == "pass")
            metrics = result.get("model_metrics") or {}
            rows.append(
                {
                    "id": case["id"],
                    "category": case["category"],
                    "passed": passed,
                    "rules": rules,
                    "judge_verdict": judge.get("verdict"),
                    "latency_ms": latency_ms,
                    "prompt_tokens": metrics.get("prompt_tokens"),
                    "completion_tokens": metrics.get("completion_tokens"),
                    "safe": result.get("safe"),
                    "answer": result.get("answer"),
                }
            )

        passed = [row for row in rows if row["passed"]]
        by_category = {}
        for row in rows:
            bucket = by_category.setdefault(row["category"], [0, 0])
            bucket[1] += 1
            if row["passed"]:
                bucket[0] += 1
        results[strategy] = {
            "total": len(rows),
            "passed": len(passed),
            "pass_rate": round(len(passed) / len(rows), 3) if rows else None,
            "by_category": by_category,
            "failed": [row["id"] for row in rows if not row["passed"]],
            "mean_latency_ms": round(
                sum(row["latency_ms"] for row in rows) / len(rows), 1
            )
            if rows
            else None,
            "total_prompt_tokens": sum(row["prompt_tokens"] or 0 for row in rows),
            "total_completion_tokens": sum(row["completion_tokens"] or 0 for row in rows),
            "wall_ms": round((time.perf_counter() - started) * 1000),
            "cases": rows,
        }
        print(
            f"{strategy:<14} {len(passed):>2}/{len(rows)} pass  "
            f"{results[strategy]['mean_latency_ms']} ms/answer  "
            f"{results[strategy]['total_prompt_tokens']} prompt tokens"
        )
        if results[strategy]["failed"]:
            print(f"  failed: {', '.join(results[strategy]['failed'])}")
        print()

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": generator.model,
        "backend": "hosted-api" if generator.use_api else "local",
        "k": args.k,
        "judge": args.judge,
        "strategies": results,
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = REPORT_DIR / f"prompts_{stamp}.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Report: {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
