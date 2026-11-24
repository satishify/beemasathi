"""Embedding backends for retrieval.

The hosted endpoint serves the same bge-small-en-v1.5 model that built the
Chroma index, so query vectors are interchangeable with the stored ones.
The local sentence-transformers model remains the fallback when no hosted
endpoint is configured.

Hosted requests are budgeted before they leave the process: the model caps
input at 512 tokens, so an over-long chunk is cut on a paragraph, sentence or
word boundary instead of being rejected. If the endpoint still reports a token
overflow, the batch is re-cut in proportion to the measured token count.
"""

import re
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import requests

from config import (
    EMBEDDING_API_BASE,
    EMBEDDING_API_BATCH,
    EMBEDDING_API_CA,
    EMBEDDING_API_KEY,
    EMBEDDING_API_KEY_FILE,
    EMBEDDING_API_MODEL,
    EMBEDDING_MAX_INPUT_CHARS,
    EMBEDDING_MAX_INPUT_TOKENS,
    EMBEDDING_MAX_RECUT_PASSES,
)

ROOT_DIR = Path(__file__).resolve().parent.parent
RETRY_DELAYS = (2, 4, 8)
RETRY_STATUSES = {429, 500, 502, 503, 504}
GIVEN_TOKENS_RE = re.compile(r"Given:\s*(\d+)")
# bge-small averages roughly four characters per token on policy prose.
_CHARS_PER_TOKEN = 4
_MIN_CUT_CHARS = 200


class EmbeddingBackendError(RuntimeError):
    pass


def _resolve(path: str) -> Path:
    candidate = Path(path).expanduser()
    return candidate if candidate.is_absolute() else ROOT_DIR / candidate


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    np.divide(vectors, norms, out=vectors, where=norms > 0)
    return vectors


def _cut_at_boundary(text: str, limit: int) -> str:
    """Trim text to at most limit chars, preferring paragraph/sentence/word ends."""
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    head = text[:limit]
    floor = max(64, min(_MIN_CUT_CHARS, limit // 4))
    for marker in ("\n\n", ". ", "\n", " "):
        cut = head.rfind(marker)
        if cut >= floor:
            return head[:cut].rstrip()
    return head


class HostedEmbeddingBackend:
    """Embed through the OpenAI-compatible endpoint, 16 texts per request."""

    def __init__(
        self,
        base: Optional[str] = None,
        model: Optional[str] = None,
        key: Optional[str] = None,
        key_file: Optional[str] = None,
        ca: Optional[str] = None,
        batch_size: Optional[int] = None,
        timeout: int = 60,
        session: Optional[requests.Session] = None,
        max_input_chars: Optional[int] = None,
        max_input_tokens: Optional[int] = None,
        max_recut_passes: Optional[int] = None,
    ):
        self.base = (EMBEDDING_API_BASE if base is None else base).rstrip("/")
        self.model = EMBEDDING_API_MODEL if model is None else model
        self.batch_size = max(
            1, int(EMBEDDING_API_BATCH if batch_size is None else batch_size)
        )
        self.timeout = timeout
        self.max_input_chars = max(
            1,
            int(
                EMBEDDING_MAX_INPUT_CHARS
                if max_input_chars is None
                else max_input_chars
            ),
        )
        self.max_input_tokens = max(
            1,
            int(
                EMBEDDING_MAX_INPUT_TOKENS
                if max_input_tokens is None
                else max_input_tokens
            ),
        )
        self.max_recut_passes = max(
            0,
            int(
                EMBEDDING_MAX_RECUT_PASSES
                if max_recut_passes is None
                else max_recut_passes
            ),
        )
        self.key = EMBEDDING_API_KEY if key is None else key
        source = EMBEDDING_API_KEY_FILE if key_file is None else key_file
        if not self.key and source:
            try:
                self.key = _resolve(source).read_text(encoding="utf-8").strip()
            except OSError as exc:
                raise EmbeddingBackendError(
                    f"Embedding key file is not readable: {source}"
                ) from exc
        self.ca = EMBEDDING_API_CA if ca is None else ca
        self.verify = str(_resolve(self.ca)) if self.ca else True
        if not self.base or not self.key:
            raise EmbeddingBackendError(
                "Hosted embeddings need BEEMASATHI_EMBED_BASE and a key."
            )
        self.session = session or requests.Session()
        self.requests_sent = 0
        self.texts_sent = 0
        self.truncated = 0
        self.recuts = 0
        self.dim: Optional[int] = None

    @property
    def name(self) -> str:
        return f"api:{self.model}"

    def stats(self) -> Dict[str, object]:
        return {
            "backend": self.name,
            "requests": self.requests_sent,
            "texts": self.texts_sent,
            "truncated": self.truncated,
            "recuts": self.recuts,
            "dim": self.dim,
        }

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        batch = [str(text) for text in texts]
        if not batch:
            return np.zeros((0, 0), dtype=np.float32)
        vectors: List[List[float]] = []
        for start in range(0, len(batch), self.batch_size):
            fitted = [
                self._fit(text) for text in batch[start:start + self.batch_size]
            ]
            vectors.extend(self._post(fitted))
        matrix = _normalize(np.asarray(vectors, dtype=np.float32))
        if matrix.size:
            if self.dim is None:
                self.dim = int(matrix.shape[1])
            elif matrix.shape[1] != self.dim:
                raise EmbeddingBackendError(
                    f"Embedding dimension changed from {self.dim} to "
                    f"{matrix.shape[1]}."
                )
        return matrix

    def _fit(self, text: str) -> str:
        limit = min(self.max_input_chars, self.max_input_tokens * _CHARS_PER_TOKEN)
        if len(text) > limit:
            text = _cut_at_boundary(text, limit)
            self.truncated += 1
        return text

    def _recut(self, batch: List[str], cut_chars: int) -> List[str]:
        cut = max(_MIN_CUT_CHARS, cut_chars)
        self.recuts += 1
        return [_cut_at_boundary(text, cut) for text in batch]

    def _request(self, batch: List[str]) -> requests.Response:
        payload = {"model": self.model, "input": batch}
        headers = {
            "Authorization": f"Bearer {self.key}",
            "Content-Type": "application/json",
        }
        last_error = None
        for delay in (0,) + RETRY_DELAYS:
            if delay:
                time.sleep(delay)
            try:
                response = self.session.post(
                    f"{self.base}/embeddings",
                    json=payload,
                    headers=headers,
                    verify=self.verify,
                    timeout=self.timeout,
                )
            except requests.RequestException as exc:
                last_error = exc
                continue
            self.requests_sent += 1
            if response.status_code in (200, 422):
                return response
            if response.status_code not in RETRY_STATUSES:
                raise EmbeddingBackendError(
                    f"Embedding API returned HTTP {response.status_code}: "
                    f"{response.text[:200]}"
                )
            last_error = EmbeddingBackendError(f"HTTP {response.status_code}")
        raise EmbeddingBackendError(
            f"Embedding API request failed after retries: {last_error}"
        )

    def _post(self, batch: List[str]) -> List[List[float]]:
        current = batch
        for attempt in range(self.max_recut_passes + 1):
            response = self._request(current)
            if response.status_code == 200:
                data = sorted(
                    response.json().get("data") or [],
                    key=lambda item: item.get("index", 0),
                )
                if len(data) != len(current):
                    raise EmbeddingBackendError(
                        f"Embedding API returned {len(data)} vectors "
                        f"for {len(current)} texts."
                    )
                self.texts_sent += len(current)
                return [item["embedding"] for item in data]
            measured = GIVEN_TOKENS_RE.search(response.text or "")
            if not measured:
                raise EmbeddingBackendError(
                    "Embedding API rejected the input as too long but did not "
                    f"report a token count: {response.text[:200]}"
                )
            if attempt == self.max_recut_passes:
                break
            longest = max((len(text) for text in current), default=0)
            ratio = self.max_input_tokens / max(1, int(measured.group(1)))
            current = self._recut(
                current, max(_MIN_CUT_CHARS, int(longest * ratio))
            )
        raise EmbeddingBackendError(
            f"Embedding input still exceeds {self.max_input_tokens} tokens after "
            f"{self.max_recut_passes} re-cut passes."
        )


class LocalEmbeddingBackend:
    """Wrap Chroma's sentence-transformers function."""

    name = "local"
    truncated = 0
    recuts = 0
    dim: Optional[int] = None

    def __init__(self, embed_fn):
        self.embed_fn = embed_fn

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        batch = list(texts)
        if not batch:
            return np.zeros((0, 0), dtype=np.float32)
        vectors = _normalize(np.asarray(self.embed_fn(batch), dtype=np.float32))
        if vectors.size and self.dim is None:
            self.dim = int(vectors.shape[1])
        return vectors

    def stats(self) -> Dict[str, object]:
        return {
            "backend": self.name,
            "truncated": 0,
            "recuts": 0,
            "dim": self.dim,
        }


def hosted_configured() -> bool:
    return bool(
        EMBEDDING_API_BASE and (EMBEDDING_API_KEY or EMBEDDING_API_KEY_FILE)
    )


def get_embedding_backend(local_ef=None):
    if hosted_configured():
        return HostedEmbeddingBackend()
    if local_ef is None:
        raise EmbeddingBackendError(
            "Local embeddings need a loaded sentence-transformers function."
        )
    return LocalEmbeddingBackend(local_ef)
