"""Retrieve policy clauses and re-rank them for the generator."""

import re
from typing import Dict, List, Optional

from rag.bm25 import get_index
from rag.embedder import PolicyVectorStore
from rag.policy_match import PRODUCT_CLASS_NAMES, match_policy, normalize
from rag.provenance import is_allowed

LINE_HINTS = (
    ("motor", r"\b(motor|car|bike|two[\s-]?wheeler|vehicle|own damage|third party|od|tp)\b"),
    ("health", r"\b(health|mediclaim|hospital|hospitalisation|hospitalization|"
               r"critical illness|opd|ped|room rent|cashless)\b"),
)

# Higher weight = more authoritative for a coverage answer.
DOC_TYPE_WEIGHT = {
    "policy_wording": 1.0,
    "endorsement": 0.9,
    "cis": 0.6,
    "prospectus": 0.5,
    "brochure": 0.3,
    "regulation": 0.4,
}

RRF_K = 60

# Shorthand people type for standard products. Each alias lists the words a
# policy name must contain to be the same product (e.g. "digital health
# arogya" still means Arogya Sanjeevani).
PRODUCT_ALIASES = {
    "arogya": ("arogya", "sanjeevani"),
    "sanjeevani": ("arogya", "sanjeevani"),
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
        """Retrieve dense and lexical candidates, fuse, then authority-rerank."""
        chunk_type = self._infer_chunk_type(query)
        pool_size = max(n_results * 3, 24)
        policy_name = policy_name or self._detect_policy(query, insurer)
        policy_name = self._resolve_wording_name(policy_name, insurer)
        search_query = self._search_query(query, policy_name, insurer)

        lists: List[List[Dict]] = [
            self.store.search(
                query=search_query,
                insurer=insurer,
                policy_name=policy_name,
                n_results=pool_size,
            )
        ]
        lexical = get_index().search(
            query=search_query,
            insurer=insurer,
            policy_name=policy_name,
            n_results=pool_size,
        )
        if lexical:
            lists.append(lexical)

        if chunk_type:
            lists.append(
                self.store.search(
                    query=search_query,
                    insurer=insurer,
                    policy_name=policy_name,
                    chunk_type=chunk_type,
                    n_results=max(n_results, 8),
                )
            )
            typed_lexical = get_index().search(
                query=search_query,
                insurer=insurer,
                policy_name=policy_name,
                chunk_type=chunk_type,
                n_results=max(n_results, 8),
            )
            if typed_lexical:
                lists.append(typed_lexical)

        merged = self._fuse([self._dedupe(results) for results in lists if results])
        if not merged and (insurer or policy_name or chunk_type):
            # An explicitly selected insurer is a hard constraint. Only the
            # policy-name and chunk-type hints are relaxed when they match nothing.
            merged = self._dedupe(
                self.store.search(query=query, insurer=insurer, n_results=pool_size)
            )

        merged = [item for item in merged if is_allowed(item.get("metadata"))]

        line = self._infer_line(query)
        if line and not insurer and not policy_name:
            merged = self._prefer_line(merged, line)

        ranked = self._rerank(merged, search_query, policy_name)
        return ranked[:n_results]

    @staticmethod
    def _prefer_line(results: List[Dict], line: str) -> List[Dict]:
        """Keep chunks from the line the question names, when any exist."""
        matching = [
            item
            for item in results
            if (item.get("metadata") or {}).get("line_of_business") == line
        ]
        return matching or results

    @staticmethod
    def _search_query(query: str, policy_name: Optional[str], insurer: Optional[str]) -> str:
        """Drop the repeated policy name so the clause words drive retrieval."""
        if not policy_name:
            return query
        drop = set(normalize(policy_name).split()) | set(normalize(insurer or "").split())
        tokens = [token for token in normalize(query).split() if token not in drop]
        return " ".join(tokens) if len(tokens) >= 2 else query

    def _fuse(self, lists: List[List[Dict]]) -> List[Dict]:
        """Reciprocal-rank fusion across dense and BM25 result lists."""
        fused: Dict[tuple, Dict] = {}
        for results in lists:
            for rank, item in enumerate(results, 1):
                meta = item.get("metadata") or {}
                key = (
                    meta.get("source"),
                    meta.get("page"),
                    (item.get("text") or "")[:160],
                )
                if key not in fused:
                    fused[key] = {**item, "rrf": 0.0}
                fused[key]["rrf"] += 1 / (RRF_K + rank)
                distance = item.get("distance")
                if distance is not None:
                    current = fused[key].get("distance")
                    fused[key]["distance"] = (
                        distance if current is None else min(current, distance)
                    )
        return list(fused.values())

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

    def _rerank(
        self, results: List[Dict], query: str, policy_name: Optional[str] = None
    ) -> List[Dict]:
        q = query.lower()
        line = self._infer_line(query)
        policy_hint = self._policy_hint(query)
        weights = self._type_weights(q)

        query_terms = [
            token
            for token in re.findall(r"[a-z0-9]{4,}", q)
            if token not in {"what", "does", "this", "that", "with", "from", "your", "have"}
        ]

        def score(item: Dict) -> float:
            meta = item.get("metadata") or {}
            text = (item.get("text") or "").lower()
            relevance = item.get("rrf")
            if relevance is None:
                distance = float(item.get("distance") or 1.0)
                relevance = 1 / (1 + distance)
            authority = weights.get(meta.get("document_type", ""), 0.35)
            line_bonus = 0.012 if line and meta.get("line_of_business") == line else 0
            policy_bonus = 0
            current_policy = meta.get("policy_name")
            policy_name_text = str(current_policy or "").lower().replace("_", " ")
            if policy_hint and policy_hint in policy_name_text:
                policy_bonus = 0.02
            if policy_name and current_policy == policy_name:
                policy_bonus = 0.03
            overlap = sum(1 for token in query_terms if token in text)
            lexical = 0.002 * overlap
            return relevance + 0.05 * authority + line_bonus + policy_bonus + lexical

        return sorted(results, key=score, reverse=True)

    @staticmethod
    def _type_weights(query: str) -> Dict[str, float]:
        """Prefer the document kind the question actually asks for."""
        if "brochure" in query or "prospectus" in query or re.search(r"\bcis\b", query):
            return {
                "brochure": 1.0,
                "prospectus": 0.9,
                "cis": 0.9,
                "policy_wording": 0.4,
                "regulation": 0.2,
            }
        if "wording" in query or "policy document" in query:
            return {
                "policy_wording": 1.0,
                "endorsement": 0.9,
                "brochure": 0.15,
                "regulation": 0.3,
            }
        return DOC_TYPE_WEIGHT

    def _infer_line(self, query: str) -> Optional[str]:
        for line, pattern in LINE_HINTS:
            if re.search(pattern, query, re.I):
                return line
        return None

    @staticmethod
    def _normalize(text: str) -> str:
        return normalize(text)

    def _detect_policy(self, query: str, insurer: Optional[str]) -> Optional[str]:
        """Return a policy name only when the question names exactly one product."""
        query_tokens = self._normalize(query).split()
        wording = self._wording_pairs()
        names = [
            (candidate_insurer, name)
            for candidate_insurer, name in get_index().policies
            if (not insurer or candidate_insurer == insurer)
            and (not wording or (candidate_insurer, name) in wording)
        ]
        matched = match_policy(query_tokens, names, insurer)
        if matched and normalize(matched) not in PRODUCT_CLASS_NAMES:
            return matched
        return self._alias_policy(query_tokens, insurer)

    def _resolve_wording_name(self, policy_name: Optional[str], insurer: Optional[str]) -> Optional[str]:
        """Map a catalog/brochure name to the loaded wording for the same product."""
        if not policy_name:
            return None
        wording = self._wording_pairs()
        if (insurer, policy_name) in wording:
            return policy_name
        names = [
            (candidate_insurer, name)
            for candidate_insurer, name in wording
            if not insurer or candidate_insurer == insurer
        ]
        return match_policy(normalize(policy_name).split(), names, insurer) or policy_name

    def _wording_pairs(self) -> set:
        """(insurer, policy_name) pairs that have a full policy wording."""
        cached = getattr(self, "_wording_cache", None)
        if cached is not None:
            return cached
        pairs = set(getattr(get_index(), "wording_policies", None) or [])
        if not pairs:
            store = getattr(self, "store", None)
            if store is not None:
                pairs = {
                    (entry.get("insurer"), entry.get("policy_name"))
                    for entry in store.get_catalog()
                    if entry.get("document_type") == "policy_wording"
                }
        self._wording_cache = pairs
        return pairs

    def _alias_policy(
        self, query_tokens: List[str], insurer: Optional[str]
    ) -> Optional[str]:
        """Resolve a product shorthand such as "arogya" to one insurer's plan."""
        if not insurer:
            return None
        tokens = set(query_tokens)
        wording = self._wording_pairs()
        for alias, required in PRODUCT_ALIASES.items():
            if alias not in tokens:
                continue
            hits = [
                name
                for candidate_insurer, name in get_index().policies
                if candidate_insurer == insurer
                and (not wording or (candidate_insurer, name) in wording)
                and all(
                    word in self._normalize(name).split() for word in required
                )
            ]
            if hits and "group" not in tokens:
                # "Arogya Sanjeevani" means the retail plan; the group variant
                # is a different product and is only matched when asked for.
                individual = [
                    name for name in hits if "group" not in self._normalize(name).split()
                ]
                hits = individual or hits
            if hits:
                return min(hits, key=len)
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
