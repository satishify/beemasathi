import json
from pathlib import Path

from rag.provenance import LEDGER_PATH, is_allowed


def test_chunk_without_source_file_is_rejected():
    assert not is_allowed({"insurer": "star_health", "policy_name": "comprehensive"})
    assert not is_allowed(None)


def test_ledgered_source_is_allowed():
    ledger = json.loads(LEDGER_PATH.read_text(encoding="utf-8"))["files"]
    name = Path(next(iter(ledger))).name
    assert is_allowed({"source_file": name})


def test_irdai_regulation_is_allowed():
    assert is_allowed(
        {"source_file": "2-IRDAI-Regulatory-Sandbox-Regulations-2025.pdf"}
    )


def test_unknown_pdf_is_rejected():
    assert not is_allowed({"source_file": "not_in_any_ledger.pdf"})
