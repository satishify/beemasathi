"""Small in-memory BM25 index over the chunk files.

The corpus is rebuilt from ``data/chunks/*.json`` so the index survives a
Chroma rebuild. No extra dependency is used; the scoring is Okapi BM25.
"""

import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from rag.provenance import is_allowed

BASE_DIR = Path(__file__).resolve().parent.parent
CHUNKS_DIR = BASE_DIR / "data" / "chunks"
TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    return [token for token in TOKEN_RE.findall(text.lower()) if len(token) > 1]


class BM25Index:
    def __init__(self, chunks: List[Dict]):
        self.metadatas = [chunk.get("metadata") or {} for chunk in chunks]
        self.texts = [chunk.get("text") or "" for chunk in chunks]
        self.doc_len: List[int] = []
        self.postings: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        self.df: Dict[str, int] = defaultdict(int)
        self.policies = sorted(
            {
                (metadata.get("insurer"), metadata.get("policy_name"))
                for metadata in self.metadatas
                if metadata.get("policy_name")
            }
        )
        self.wording_policies = sorted(
            {
                (metadata.get("insurer"), metadata.get("policy_name"))
                for metadata in self.metadatas
                if metadata.get("policy_name")
                and metadata.get("document_type") == "policy_wording"
            }
        )

        for doc_id, (text, metadata) in enumerate(zip(self.texts, self.metadatas)):
            # Index the policy name too: questions name the product far more
            # often than the wording repeats it inside a clause.
            header = " ".join(
                str(metadata.get(key) or "").replace("_", " ")
                for key in ("insurer", "policy_name", "document_type")
            )
            tokens = tokenize(f"{header} {text}")
            self.doc_len.append(len(tokens))
            counts: Dict[str, int] = defaultdict(int)
            for token in tokens:
                counts[token] += 1
            for token, count in counts.items():
                self.postings[token].append((doc_id, count))
                self.df[token] += 1

        self.total_docs = len(self.texts)
        self.avg_len = (sum(self.doc_len) / self.total_docs) if self.total_docs else 0.0
        self.k1 = 1.5
        self.b = 0.75

    def _matches(self, meta: Dict, insurer: Optional[str], policy_name: Optional[str], chunk_type: Optional[str]) -> bool:
        if insurer and meta.get("insurer") != insurer:
            return False
        if policy_name and meta.get("policy_name") != policy_name:
            return False
        if chunk_type and meta.get("chunk_type") != chunk_type:
            return False
        return True

    def search(
        self,
        query: str,
        insurer: Optional[str] = None,
        policy_name: Optional[str] = None,
        chunk_type: Optional[str] = None,
        n_results: int = 8,
    ) -> List[Dict]:
        terms = list(dict.fromkeys(tokenize(query)))
        if not terms or not self.total_docs:
            return []

        # A filter mask avoids scoring documents that cannot be returned.
        mask = None
        if insurer or policy_name or chunk_type:
            mask = [
                self._matches(meta, insurer, policy_name, chunk_type)
                for meta in self.metadatas
            ]

        scores: Dict[int, float] = defaultdict(float)
        for term in terms:
            postings = self.postings.get(term)
            if not postings:
                continue
            idf = math.log(
                1 + (self.total_docs - self.df[term] + 0.5) / (self.df[term] + 0.5)
            )
            for doc_id, tf in postings:
                if mask is not None and not mask[doc_id]:
                    continue
                norm = self.k1 * (
                    1 - self.b + self.b * self.doc_len[doc_id] / (self.avg_len or 1)
                )
                scores[doc_id] += idf * (tf * (self.k1 + 1)) / (tf + norm)

        ranked = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
        results = []
        for doc_id, score in ranked[:n_results]:
            results.append(
                {
                    "text": self.texts[doc_id],
                    "metadata": self.metadatas[doc_id],
                    "distance": round(1 / (1 + score), 4),
                    "bm25_score": round(score, 4),
                }
            )
        return results


def load_chunks() -> List[Dict]:
    chunks: List[Dict] = []
    for path in sorted(CHUNKS_DIR.glob("*.json")):
        if path.name == "ingest_index.json":
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, list):
            continue
        for chunk in payload:
            text = chunk.get("text")
            if text and is_allowed(chunk.get("metadata")):
                chunks.append(chunk)
    return chunks


_CACHE: Optional[BM25Index] = None


def get_index() -> BM25Index:
    global _CACHE
    if _CACHE is None:
        _CACHE = BM25Index(load_chunks())
    return _CACHE
