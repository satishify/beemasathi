"""Retrieve policy clauses and re-rank them for the generator."""

import re
from typing import Dict, List, Optional

from rag.embedder import PolicyVectorStore

LINE_HINTS = (
    ("motor", r"\b(motor|car|bike|two[\s-]?wheeler|vehicle|own damage|third party|od|tp)\b"),
    ("health", r"\b(health|mediclaim|hospital|hospitalisation|hospitalization|"
               r"critical illness|opd|ped|room rent|cashless)\b"),
)

DOC_TYPE_RANK = {
    "policy_wording": 0,
    "cis": 1,
    "prospectus": 2,
    "brochure": 3,
    "regulation": 4,
}


class PolicyRetriever:
    def __init__(self):
        self.store = PolicyVectorStore()

    def retrieve(
        self,
        query: str,
        insurer: str = None,
        policy_name: str = None,
        n_results: int = 8,
    ) -> List[Dict]:
        """Retrieve relevant policy clauses, then re-rank the shortlist."""
        chunk_type = self._infer_chunk_type(query)
        pool_size = max(n_results * 3, 16)

        primary = self.store.search(
            query=query,
            insurer=insurer,
            policy_name=policy_name,
            n_results=pool_size,
        )
        typed = []
        if chunk_type:
            typed = self.store.search(
                query=query,
                insurer=insurer,
                policy_name=policy_name,
                chunk_type=chunk_type,
                n_results=max(n_results, 8),
            )

        merged = self._dedupe(typed + primary)
        if not merged and (insurer or policy_name or chunk_type):
            merged = self.store.search(query=query, n_results=pool_size)

        ranked = self._rerank(merged, query)
        return ranked[:n_results]

    def _dedupe(self, results: List[Dict]) -> List[Dict]:
        seen = set()
        unique = []
        for item in results:
            meta = item.get("metadata") or {}
            key = (
                meta.get("source"),
                meta.get("page"),
                item.get("text", "")[:160],
            )
            if key in seen:
                continue
            seen.add(key)
            unique.append(item)
        return unique

    def _rerank(self, results: List[Dict], query: str) -> List[Dict]:
        q = query.lower()
        line = self._infer_line(query)
        policy_hint = self._policy_hint(query)

        query_terms = [
            token
            for token in re.findall(r"[a-z0-9]{4,}", q)
            if token not in {"what", "does", "this", "that", "with", "from", "your", "have"}
        ]

        def score(item: Dict) -> tuple:
            meta = item.get("metadata") or {}
            text = (item.get("text") or "").lower()
            distance = float(item.get("distance") or 1.0)
            line_bonus = 0
            if line and meta.get("line_of_business") == line:
                line_bonus = -0.08
            type_rank = DOC_TYPE_RANK.get(meta.get("document_type", ""), 5)
            policy_bonus = 0
            policy_name = str(meta.get("policy_name", "")).lower().replace("_", " ")
            if policy_hint and policy_hint in policy_name:
                policy_bonus = -0.12
            overlap = sum(1 for token in query_terms if token in text)
            lexical = -0.015 * overlap
            return (distance + line_bonus + policy_bonus + lexical, type_rank)

        return sorted(results, key=score)

    def _infer_line(self, query: str) -> Optional[str]:
        for line, pattern in LINE_HINTS:
            if re.search(pattern, query, re.I):
                return line
        return None

    def _policy_hint(self, query: str) -> str:
        q = query.lower()
        phrases = (
            "critical illness",
            "health guard",
            "everyday health",
            "liability only",
            "medicare",
            "family optima",
            "family health",
            "arogya sanjeevani",
        )
        for phrase in phrases:
            if phrase in q:
                return phrase
        return ""

    def _infer_chunk_type(self, query: str) -> Optional[str]:
        q = query.lower()
        if any(w in q for w in ["room rent", "room category", "ac room", "single room"]):
            return "room_rent"
        if any(w in q for w in ["pre-existing", "ped", "waiting period", "diabetes", "bp", "hypertension"]):
            return "waiting_period"
        if any(w in q for w in ["co-pay", "copay", "deductible", "co payment"]):
            return "co_pay"
        if any(w in q for w in ["not covered", "exclusion", "will not pay", "own damage"]):
            return "exclusion"
        if any(w in q for w in ["cashless", "network", "tpa", "hospital list", "ppn"]):
            return "cashless"
        if any(w in q for w in ["claim", "rejection", "documents", "intimation", "form"]):
            return "claim_process"
        if any(w in q for w in ["sum insured", "coverage", "policy limit"]):
            return "coverage"
        return None


if __name__ == "__main__":
    retriever = PolicyRetriever()
    for q in (
        "What is the room rent cap?",
        "Does Digit private car liability only cover own damage?",
    ):
        print(f"\n=== Query: {q} ===")
        for row in retriever.retrieve(q)[:3]:
            meta = row["metadata"]
            print(
                f"  {meta.get('insurer')} | {meta.get('policy_name')} | "
                f"{meta.get('document_type')} | d={row.get('distance'):.3f}"
            )
