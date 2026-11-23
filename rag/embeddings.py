"""Embedding backends for retrieval.

The hosted endpoint serves the same bge-small-en-v1.5 model that built the
Chroma index, so query vectors are interchangeable with the stored ones.
The local sentence-transformers model remains the fallback when no hosted
endpoint is configured.
"""

import time
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import requests

from config import (
    EMBEDDING_API_BASE,
    EMBEDDING_API_BATCH,
    EMBEDDING_API_CA,
    EMBEDDING_API_KEY,
    EMBEDDING_API_KEY_FILE,
    EMBEDDING_API_MODEL,
)

ROOT_DIR = Path(__file__).resolve().parent.parent
RETRY_DELAYS = (2, 4, 8)
RETRY_STATUSES = {429, 500, 502, 503, 504}


class EmbeddingBackendError(RuntimeError):
    pass


def _resolve(path: str) -> Path:
    candidate = Path(path).expanduser()
    return candidate if candidate.is_absolute() else ROOT_DIR / candidate


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    np.divide(vectors, norms, out=vectors, where=norms > 0)
    return vectors


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
    ):
        self.base = (EMBEDDING_API_BASE if base is None else base).rstrip("/")
        self.model = EMBEDDING_API_MODEL if model is None else model
        self.batch_size = max(
            1, int(EMBEDDING_API_BATCH if batch_size is None else batch_size)
        )
        self.timeout = timeout
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
        self.session = requests.Session()

    @property
    def name(self) -> str:
        return f"api:{self.model}"

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        batch = [str(text) for text in texts]
        if not batch:
            return np.zeros((0, 0), dtype=np.float32)
        vectors: List[List[float]] = []
        for start in range(0, len(batch), self.batch_size):
            vectors.extend(self._post(batch[start:start + self.batch_size]))
        return _normalize(np.asarray(vectors, dtype=np.float32))

    def _post(self, batch: List[str]) -> List[List[float]]:
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
            if response.status_code == 200:
                data = sorted(
                    response.json().get("data") or [],
                    key=lambda item: item.get("index", 0),
                )
                if len(data) != len(batch):
                    raise EmbeddingBackendError(
                        f"Embedding API returned {len(data)} vectors "
                        f"for {len(batch)} texts."
                    )
                return [item["embedding"] for item in data]
            if response.status_code not in RETRY_STATUSES:
                raise EmbeddingBackendError(
                    f"Embedding API returned HTTP {response.status_code}: "
                    f"{response.text[:200]}"
                )
            last_error = EmbeddingBackendError(f"HTTP {response.status_code}")
        raise EmbeddingBackendError(
            f"Embedding API request failed after retries: {last_error}"
        )


class LocalEmbeddingBackend:
    """Wrap Chroma's sentence-transformers function."""

    name = "local"

    def __init__(self, embed_fn):
        self.embed_fn = embed_fn

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        batch = list(texts)
        if not batch:
            return np.zeros((0, 0), dtype=np.float32)
        vectors = np.asarray(self.embed_fn(batch), dtype=np.float32)
        return _normalize(vectors)


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
