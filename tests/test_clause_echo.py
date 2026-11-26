from rag.generator import BimaSaathiGenerator


def _bot():
    return BimaSaathiGenerator.__new__(BimaSaathiGenerator)


def test_clause_header_echo_is_stripped():
    answer = (
        "Deciding Clause: [2] digit | digit_two_wheeler_package_policy_long_term | "
        'Page 5 | policy_wording | page 5 "Policy may be cancelled at the option of '
        'the insured at any time during the term, by informing Us." Answer: '
        "- The clause allows cancellation at any time by the insured."
    )

    cleaned = _bot()._sanitize_answer(answer)

    assert "Deciding Clause" not in cleaned
    assert "policy_wording" not in cleaned
    assert "[2]" not in cleaned
    assert cleaned.startswith("- The clause allows cancellation")


def test_bold_echo_labels_are_stripped():
    answer = (
        "**Deciding Clause**: [1] digit | digit_two_wheeler | Page 2 | "
        "policy_wording | page 2\n**Answer:** Third-party cover can be added later."
    )

    cleaned = _bot()._sanitize_answer(answer)

    assert cleaned == "Third-party cover can be added later."


def test_plain_answer_is_untouched():
    answer = "Yes. You can add third-party cover later.\n- No deadline is stated."

    assert _bot()._sanitize_answer(answer) == answer


def test_repeated_sentences_are_dropped():
    answer = (
        "Room rent is capped at 1% of sum insured. "
        "PED waits 36 months. "
        "Room rent is capped at 1% of sum insured. "
        "PED waits 36 months. "
        "Room rent is capped at 1% of sum insured."
    )

    cleaned = _bot()._sanitize_answer(answer)

    assert cleaned == "Room rent is capped at 1% of sum insured. PED waits 36 months."
