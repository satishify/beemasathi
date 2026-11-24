import numpy as np
import pytest

from rag.embeddings import (
    EmbeddingBackendError,
    HostedEmbeddingBackend,
    LocalEmbeddingBackend,
    _cut_at_boundary,
)
from rag.index_manifest import read_manifest, verify_manifest, write_manifest


class _Response:
    def __init__(self, status_code, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text

    def json(self):
        return self._payload


class _Session:
    """Record requests and answer with a scripted list of responses."""

    def __init__(self, responses, dim=4):
        self.responses = list(responses)
        self.requests = []
        self.dim = dim

    def post(self, url, json=None, headers=None, verify=None, timeout=None):
        self.requests.append(json)
        if not self.responses:
            return self._ok(json["input"])
        response = self.responses.pop(0)
        if callable(response):
            return response(json["input"])
        return response

    def _ok(self, batch):
        data = [
            {"index": i, "embedding": [float(i + 1)] * self.dim}
            for i in range(len(batch))
        ]
        return _Response(200, {"data": data})


def _backend(session, **kwargs):
    return HostedEmbeddingBackend(
        base="https://embed.example/v1",
        model="test-model",
        key="test-key",
        session=session,
        **kwargs,
    )


def test_cut_at_boundary_prefers_paragraph_then_word():
    text = ("A" * 90) + "\n\n" + "word " * 400
    cut = _cut_at_boundary(text, 200)
    assert len(cut) <= 200
    assert cut == "A" * 90
    assert _cut_at_boundary("tiny", 120) == "tiny"


def test_long_text_is_fit_to_budget_before_request():
    session = _Session([])
    backend = _backend(
        session, max_input_chars=400, max_input_tokens=512, batch_size=2
    )
    text = "sentence about room rent. " * 200
    backend.encode([text])
    sent = session.requests[0]["input"][0]
    assert len(sent) <= 400
    assert backend.truncated == 1


def test_422_token_overflow_recuts_and_succeeds():
    calls = {"n": 0}

    def scripted(batch):
        calls["n"] += 1
        if calls["n"] == 1:
            return _Response(422, text="Given: 900 tokens, limit is 512")
        return _Response(
            200,
            {
                "data": [
                    {"index": i, "embedding": [1.0, 0.0, 0.0]}
                    for i in range(len(batch))
                ]
            },
        )

    session = _Session([scripted, scripted])
    backend = _backend(session, batch_size=1, max_recut_passes=2)
    vectors = backend.encode(["x " * 500])
    assert vectors.shape == (1, 3)
    assert backend.recuts == 1
    assert len(session.requests) == 2
    assert len(session.requests[1]["input"][0]) < len(session.requests[0]["input"][0])


def test_422_without_token_count_raises():
    session = _Session([_Response(422, text="input too long")])
    backend = _backend(session)
    with pytest.raises(EmbeddingBackendError, match="token count"):
        backend.encode(["text"])


def test_422_after_all_recut_passes_raises():
    session = _Session(
        [_Response(422, text="Given: 5000") for _ in range(3)]
    )
    backend = _backend(session, max_recut_passes=2)
    with pytest.raises(EmbeddingBackendError, match="re-cut passes"):
        backend.encode(["text"])


def test_dimension_change_is_rejected():
    session = _Session([])
    session.dim = 4
    backend = _backend(session, batch_size=1)
    backend.encode(["first"])
    session.dim = 5
    with pytest.raises(EmbeddingBackendError, match="dimension changed"):
        backend.encode(["second"])


def test_vector_count_mismatch_is_rejected():
    session = _Session([_Response(200, {"data": [{"index": 0, "embedding": [1.0]}]})])
    backend = _backend(session)
    with pytest.raises(EmbeddingBackendError, match="returned 1 vectors"):
        backend.encode(["a", "b"])


def test_local_backend_records_dimension():
    backend = LocalEmbeddingBackend(lambda texts: [[1.0, 0.0]] * len(texts))
    vectors = backend.encode(["a", "b"])
    assert vectors.shape == (2, 2)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)
    assert backend.dim == 2
    assert backend.stats()["dim"] == 2


def test_manifest_round_trip_and_verification(tmp_path):
    path = tmp_path / "index_manifest.json"
    write_manifest(
        path,
        model="BAAI/bge-small-en-v1.5",
        dim=384,
        chunks=10,
        params={"chunk_max_chars": 1800},
    )
    payload = read_manifest(path)
    assert payload["chunks"] == 10
    assert verify_manifest(path, model="BAAI/bge-small-en-v1.5", dim=384) == []
    assert "queries use" in verify_manifest(path, model="other")[0]
    assert "dimension" in verify_manifest(path, model="BAAI/bge-small-en-v1.5", dim=768)[0]


def test_missing_manifest_is_reported_not_raised(tmp_path):
    problems = verify_manifest(tmp_path / "absent.json", model="any")
    assert problems == [
        "index manifest is missing; the index cannot be traced to a model"
    ]
