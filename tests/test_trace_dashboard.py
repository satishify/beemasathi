import json

from observability.dashboard import render


def _write(directory, rows):
    directory.mkdir()
    (directory / "2026-11-26.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows), encoding="utf-8"
    )


def test_dashboard_renders_counts_and_escapes_the_preview(tmp_path):
    traces = tmp_path / "traces"
    _write(
        traces,
        [
            {
                "ts": "2026-11-26T10:00:00+00:00",
                "question_preview": "<script>alert(1)</script>",
                "intent": "coverage",
                "plan": {"insurers": ["max_bupa"]},
                "retrieved": 4,
                "verified": True,
                "total_ms": 1200,
                "model": "test-model",
                "safe": True,
                "sources": [{"policy_name": "health_plus"}],
            },
            {
                "ts": "2026-11-26T11:00:00+00:00",
                "question_preview": "Compare two plans",
                "intent": "comparison",
                "plan": {"insurers": ["a", "b"]},
                "retrieved": 6,
                "verified": False,
                "total_ms": 900,
                "model": "test-model",
                "safe": True,
                "sources": [],
            },
        ],
    )

    page = render(traces, tmp_path / "dashboard.html").read_text(encoding="utf-8")

    assert "Requests traced</span><span class=\"v\">2" in page
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in page
    assert "<script>alert(1)</script>" not in page
    assert "health_plus" in page
    assert "comparison" in page


def test_dashboard_handles_an_empty_log(tmp_path):
    traces = tmp_path / "traces"
    traces.mkdir()

    page = render(traces, tmp_path / "dashboard.html").read_text(encoding="utf-8")

    assert "No data yet" in page or "No sources recorded" in page
