from agents.critic import (
    resolve_authority,
    strip_unbacked_family_limits,
    verify_answer,
)


def test_duration_and_percent_equivalence():
    chunks = [{"text": "Specified diseases are excluded for 2 years. Co-pay is 5% of each claim."}]
    result = verify_answer("The wait is 24 months and the co-pay is 5 per cent.", chunks)
    assert result["verified"] is True


def test_unsupported_uin_is_flagged():
    chunks = [{"text": "The co-pay is 5% of each claim."}]
    result = verify_answer("Co-pay is 5%. UIN TATHLIP21292V022021.", chunks)
    assert result["verified"] is False
    assert "TATHLIP21292V022021" in result["unsupported"]


def test_authority_drops_brochure_when_wording_present():
    chunks = [
        {"text": "brochure", "metadata": {"policy_name": "p", "document_type": "brochure"}},
        {"text": "wording", "metadata": {"policy_name": "p", "document_type": "policy_wording"}},
        {"text": "other", "metadata": {"policy_name": "q", "document_type": "brochure"}},
    ]
    kept = resolve_authority(chunks)
    assert [chunk["text"] for chunk in kept] == ["wording", "other"]


def test_brochure_intent_keeps_brochure():
    chunks = [
        {"text": "brochure", "metadata": {"policy_name": "p", "document_type": "brochure"}},
        {"text": "wording", "metadata": {"policy_name": "p", "document_type": "policy_wording"}},
    ]
    assert resolve_authority(chunks, prefer_brochure=True) == chunks


def test_family_size_echo_is_removed_without_clause_support():
    chunks = [{"text": "The sum insured is shared among insured members."}]
    answer = "The plan covers a family of 4 including 2 kids. The sum insured is shared."
    cleaned = strip_unbacked_family_limits(answer, chunks)
    assert "family of 4" not in cleaned
    assert "does not state a family-size limit" in cleaned
    assert "The sum insured is shared." in cleaned


def test_family_size_claim_is_kept_when_wording_supports_it():
    chunks = [{"text": "This policy allows a family of 4 members under one sum insured."}]
    answer = "The plan covers a family of 4 under one sum insured."
    assert strip_unbacked_family_limits(answer, chunks) == answer
