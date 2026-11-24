"""Per-request JSONL traces with redaction and a small summary CLI."""

import hashlib
import json
import os
import re
import statistics
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

BASE_DIR = Path(__file__).resolve().parent.parent
TRACE_DIR = BASE_DIR / "logs" / "traces"
ENABLED = os.getenv("BEEMASATHI_TRACE", "1") != "0"

REDACTIONS = (
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[email]"),
    (re.compile(r"\b(?:\+91[\s-]?)?[6-9]\d{9}\b"), "[phone]"),
    (re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b"), "[id]"),
    (re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"), "[pan]"),
    (re.compile(r"\b(?:\d[ -]?){13,16}\b"), "[card]"),
    (re.compile(r"\bsk-[A-Za-z0-9]{12,}\b"), "[secret]"),
    (re.compile(r"\bBearer\s+\S+", re.I), "[secret]"),
)


def redact(text: str) -> str:
    for pattern, replacement in REDACTIONS:
        text = pattern.sub(replacement, text or "")
    return text


def _hash(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:16]


class Trace:
    def __init__(self, question: str, **fields: Any):
        self.trace_id = uuid.uuid4().hex
        self.record: Dict[str, Any] = {
            "trace_id": self.trace_id,
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": "rag_answer",
            "question_hash": _hash(question),
            "question_preview": redact(question)[:160],
            **fields,
        }

    def add(self, **fields: Any) -> None:
        self.record.update(fields)

    def record_result(self, result: Dict, total_ms: float) -> None:
        metrics = result.get("model_metrics") or {}
        self.record.update(
            {
                "intent": (result.get("plan") or {}).get("intent"),
                "plan": result.get("plan"),
                "retrieved": result.get("retrieved"),
                "safe": result.get("safe"),
                "verified": (result.get("verification") or {}).get("verified"),
                "unsupported": (result.get("verification") or {}).get("unsupported"),
                "rejection_risk": result.get("rejection_risk"),
                "guard": result.get("guard"),
                "model": metrics.get("model"),
                "prompt_tokens": metrics.get("prompt_tokens"),
                "completion_tokens": metrics.get("completion_tokens"),
                "total_ms": round(total_ms, 1),
                "sources": [
                    {
                        "source": source.get("source"),
                        "policy_name": source.get("policy_name"),
                        "document_type": source.get("document_type"),
                        "page": source.get("page"),
                    }
                    for source in (result.get("sources") or [])[:8]
                ],
            }
        )

    def record_error(self, exc: Exception) -> None:
        self.add(error=f"{type(exc).__name__}: {redact(str(exc))[:200]}")

    def emit(self) -> None:
        if not ENABLED:
            return
        try:
            day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            TRACE_DIR.mkdir(parents=True, exist_ok=True)
            with (TRACE_DIR / f"{day}.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(self.record, ensure_ascii=False) + "\n")
        except OSError:
            pass


def _percentile(values: List[float], fraction: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round(fraction * (len(ordered) - 1))))
    return round(ordered[index], 1)


def summarize(directory: Path = TRACE_DIR) -> Dict[str, Any]:
    records = []
    for path in sorted(directory.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    latencies = [row.get("total_ms") for row in records if row.get("total_ms")]
    return {
        "requests": len(records),
        "errors": sum(1 for row in records if row.get("error")),
        "unsafe": sum(1 for row in records if row.get("safe") is False),
        "unverified": sum(1 for row in records if row.get("verified") is False),
        "mean_ms": round(statistics.mean(latencies), 1) if latencies else None,
        "p50_ms": _percentile(latencies, 0.5),
        "p95_ms": _percentile(latencies, 0.95),
        "prompt_tokens": sum(row.get("prompt_tokens") or 0 for row in records),
        "completion_tokens": sum(row.get("completion_tokens") or 0 for row in records),
        "intents": {
            intent: sum(1 for row in records if row.get("intent") == intent)
            for intent in sorted({row.get("intent") for row in records if row.get("intent")})
        },
    }


if __name__ == "__main__":
    print(json.dumps(summarize(), indent=2))
