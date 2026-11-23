import json
from pathlib import Path

GOLD = Path(__file__).parent.parent / "eval" / "gold_cases.json"


def _cases():
    return json.loads(GOLD.read_text())["cases"]


def test_gold_set_is_large_enough():
    assert len(_cases()) == 100


def test_ids_are_unique_and_questions_present():
    cases = _cases()
    ids = [case["id"] for case in cases]
    assert len(ids) == len(set(ids))
    assert all(case["question"].strip() for case in cases)


def test_every_case_has_a_checkable_expectation():
    for case in _cases():
        assert (
            case.get("must_include_any")
            or case.get("must_not_include")
            or case.get("expect_refuse")
            or case.get("expect_block")
            or case.get("gold_fact")
        ), case["id"]


def test_blocked_cases_forbid_dangerous_output():
    for case in _cases():
        if case.get("expect_block"):
            assert case.get("must_not_include"), case["id"]


def test_redteam_coverage():
    redteam = [case for case in _cases() if case["id"].startswith("redteam_")]
    assert len(redteam) >= 8
    joined = " ".join(case["question"].lower() for case in redteam)
    for topic in ("system prompt", "api key", "pretend", "knowledge base", "aadhaar"):
        assert topic in joined, topic
