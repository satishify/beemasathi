import numpy as np
import pytest

from rag import embeddings as embeddings_module


class _Response:
    def __init__(self, status_code, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text

    def json(self):
        return self._payload


def _payload(count):
    return {
        "data": [
            {"index": index, "embedding": [float(index + 1), 1.0]}
            for index in range(count)
        ]
    }


def test_local_backend_normalizes_vectors():
    backend = embeddings_module.LocalEmbeddingBackend(
        lambda texts: [[3.0, 4.0] for _ in texts]
    )
    vectors = backend.encode(["one", "two"])
    assert vectors.shape == (2, 2)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)


def test_hosted_backend_batches_at_sixteen(monkeypatch):
    batches = []

    def post(url, json=None, headers=None, verify=None, timeout=None):
        batches.append(len(json["input"]))
        return _Response(200, _payload(len(json["input"])))

    backend = embeddings_module.HostedEmbeddingBackend(
        base="https://embedding.test/v1", key="test-key", ca="", batch_size=16
    )
    monkeypatch.setattr(backend.session, "post", post)
    vectors = backend.encode([f"text-{i}" for i in range(20)])

    assert batches == [16, 4]
    assert vectors.shape == (20, 2)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)


def test_hosted_backend_retries_capacity_errors(monkeypatch):
    responses = [_Response(503, text="capacity"), _Response(200, _payload(2))]
    monkeypatch.setattr(embeddings_module.time, "sleep", lambda seconds: None)

    backend = embeddings_module.HostedEmbeddingBackend(
        base="https://embedding.test/v1", key="test-key", ca="", batch_size=2
    )
    monkeypatch.setattr(backend.session, "post", lambda *a, **k: responses.pop(0))
    vectors = backend.encode(["one", "two"])

    assert vectors.shape == (2, 2)
    assert responses == []


def test_hosted_backend_stops_on_auth_error(monkeypatch):
    backend = embeddings_module.HostedEmbeddingBackend(
        base="https://embedding.test/v1", key="test-key", ca="", batch_size=1
    )
    monkeypatch.setattr(
        backend.session,
        "post",
        lambda *a, **k: _Response(401, text="A valid key for this endpoint is required"),
    )
    with pytest.raises(embeddings_module.EmbeddingBackendError, match="HTTP 401"):
        backend.encode(["one"])


def test_hosted_backend_reads_key_file(tmp_path):
    key_file = tmp_path / "key.txt"
    key_file.write_text("file-key\n", encoding="utf-8")
    backend = embeddings_module.HostedEmbeddingBackend(
        base="https://embedding.test/v1",
        key="",
        key_file=str(key_file),
        ca="",
        batch_size=1,
    )
    assert backend.key == "file-key"


def test_hosted_backend_requires_a_key():
    with pytest.raises(embeddings_module.EmbeddingBackendError):
        embeddings_module.HostedEmbeddingBackend(
            base="https://embedding.test/v1", key="", key_file="", ca=""
        )
