"""Measure retrieval only: doc@k, supporting-clause@k, and mean rank.

This never calls the LLM, so it is the fast feedback loop for retrieval
changes. Use ``eval/judge.py`` for answer quality.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval.judge import _evidence_hit, _recall_hit
from rag.retriever import PolicyRetriever

GOLD_PATH = ROOT / "eval" / "gold_cases.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval-only evaluation.")
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--case", action="append", help="Run only these case ids.")
    parser.add_argument("--json-out", type=Path, help="Write per-case results to this file.")
    args = parser.parse_args()

    cases = json.loads(GOLD_PATH.read_text(encoding="utf-8"))["cases"]
    retriever = PolicyRetriever()

    rows = []
    for case in cases:
        if args.case and case["id"] not in args.case:
            continue
        chunks = retriever.retrieve(
            query=case["question"], insurer=case.get("insurer"), n_results=args.k
        )
        doc_hit, doc_rank = _recall_hit(chunks, case, args.k)
        ev_hit, ev_rank = _evidence_hit(chunks, case, args.k)
        rows.append(
            {
                "id": case["id"],
                "doc_hit": doc_hit,
                "doc_rank": doc_rank,
                "evidence_hit": ev_hit,
                "evidence_rank": ev_rank,
            }
        )
        label = "MISS" if doc_hit is False else "ok"
        ev_label = "ev-miss" if ev_hit is False else (f"ev@{ev_rank}" if ev_hit else "")
        print(f"[{label:4}] {case['id']:<46} {ev_label}")

    doc_labeled = [row for row in rows if row["doc_hit"] is not None]
    doc_hits = [row for row in doc_labeled if row["doc_hit"]]
    ev_labeled = [row for row in rows if row["evidence_hit"] is not None]
    ev_hits = [row for row in ev_labeled if row["evidence_hit"]]

    def mean(values):
        return round(sum(values) / len(values), 1) if values else None

    summary = {
        "k": args.k,
        "doc_labeled": len(doc_labeled),
        "doc_hits": len(doc_hits),
        "doc_recall": round(len(doc_hits) / len(doc_labeled), 3) if doc_labeled else None,
        "mean_doc_rank": mean([row["doc_rank"] for row in doc_hits]),
        "evidence_labeled": len(ev_labeled),
        "evidence_hits": len(ev_hits),
        "evidence_recall": round(len(ev_hits) / len(ev_labeled), 3) if ev_labeled else None,
        "mean_evidence_rank": mean([row["evidence_rank"] for row in ev_hits]),
        "doc_misses": [row["id"] for row in doc_labeled if not row["doc_hit"]],
        "evidence_misses": [row["id"] for row in ev_labeled if not row["evidence_hit"]],
    }
    print("\n" + "=" * 72)
    print(
        f"doc@{args.k}={summary['doc_recall']} ({summary['doc_hits']}/{summary['doc_labeled']})  "
        f"evidence@{args.k}={summary['evidence_recall']} "
        f"({summary['evidence_hits']}/{summary['evidence_labeled']})  "
        f"mean doc rank={summary['mean_doc_rank']}  "
        f"mean evidence rank={summary['mean_evidence_rank']}"
    )
    if summary["doc_misses"]:
        print("Doc misses:", ", ".join(summary["doc_misses"]))
    print("=" * 72)

    if args.json_out:
        summary["rows"] = rows
        args.json_out.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
