"""Question shape: multi-part questions decompose into a table; one-part stays prose."""

from rag.generator import (
    BimaSaathiGenerator,
    asked_aspects,
    compare_mode,
    multi_part_question,
)


def _bot():
    bot = BimaSaathiGenerator.__new__(BimaSaathiGenerator)
    bot.model = "test-model"
    bot.use_api = False
    return bot


def _chunk(insurer, text, page):
    return {
        "text": text,
        "metadata": {
            "insurer": insurer,
            "policy_name": f"{insurer}_health",
            "source": f"{insurer}_demo",
            "source_file": f"{insurer}_demo.pdf",
            "section": f"Sec. {page}",
            "page": page,
            "document_type": "policy_wording",
        },
    }


def test_multi_part_needs_two_aspects_joined_together():
    assert multi_part_question("What are the waiting period and co-pay for this plan?")
    assert multi_part_question("Show the room rent, exclusions, and sub-limits.")
    assert multi_part_question("Does it cover maternity? What is the waiting period?")
    assert not multi_part_question("What is the room rent cap for this plan?")
    assert not multi_part_question(
        "I made no claim last year. What No Claim Bonus does the wording show on renewal?"
    )
    assert not multi_part_question(
        "What grace period applies for renewal of Bajaj Arogya Sanjeevani?"
    )


def test_asked_aspects_are_ordered_and_deduplicated():
    aspects = asked_aspects("What are the waiting period, co-pay and exclusions?")

    assert aspects == ["waiting period", "co-pay", "exclusions"]
    assert compare_mode("Compare the room rent limit for both plans.") == "side_by_side"
    assert compare_mode("What constraints differ between the two plans?") == "differences"


def test_multi_part_question_gets_a_table_and_decompose_prompt(monkeypatch):
    bot = _bot()
    captured = {}

    def fake_call(prompt, messages=None):
        captured["prompt"] = prompt
        return "| Aspect | What the wording says | Where |\n| --- | --- | --- |", {
            "backend": "test"
        }

    monkeypatch.setattr(bot, "_call_model", fake_call)

    result = bot.generate(
        [_chunk("hdfc_ergo", "Waiting period is 2 years. Co-pay is 10%.", 12)],
        "What are the waiting period and co-pay for HDFC Ergo Optima Super?",
    )

    assert "| Aspect | What the wording says | Where |" in captured["prompt"]
    assert "Split the customer's question into the separate parts" in captured["prompt"]
    # The instruction names every asked aspect, so none is silently dropped.
    assert "The aspects asked, in order: waiting period, co-pay." in captured["prompt"]
    assert result["safe"] is True


def test_multi_part_prompt_ignores_chat_history(monkeypatch):
    bot = _bot()
    captured = {}

    def fake_call(prompt, messages=None):
        captured["prompt"] = prompt
        return "| Aspect | What the wording says | Where |\n| --- | --- | --- |", {
            "backend": "test"
        }

    monkeypatch.setattr(bot, "_call_model", fake_call)

    bot.generate(
        [_chunk("hdfc_ergo", "Waiting period is 2 years. Co-pay is 10%.", 12)],
        "What are the waiting period and co-pay for HDFC Ergo Optima Super?",
        history=[
            {"role": "user", "content": "HISTORY-MARKER-42"},
            {"role": "assistant", "content": "| Aspect | What the wording says |"},
        ],
    )

    assert "HISTORY-MARKER-42" not in captured["prompt"]


def test_one_part_question_keeps_the_plain_prompt(monkeypatch):
    bot = _bot()
    captured = {}

    def fake_call(prompt, messages=None):
        captured["prompt"] = prompt
        return "The room rent cap is 1% of sum insured.", {"backend": "test"}

    monkeypatch.setattr(bot, "_call_model", fake_call)

    bot.generate(
        [_chunk("hdfc_ergo", "Room rent is capped at 1% of sum insured.", 12)],
        "What is the room rent cap for HDFC Ergo Optima Super?",
    )

    assert "| Aspect |" not in captured["prompt"]
    assert "Split the customer's question" not in captured["prompt"]


def test_explicit_table_request_keeps_its_own_contract(monkeypatch):
    bot = _bot()
    captured = {}

    def fake_call(prompt, messages=None):
        captured["prompt"] = prompt
        return "| Fact | Value |\n| --- | --- |", {"backend": "test"}

    monkeypatch.setattr(bot, "_call_model", fake_call)

    bot.generate(
        [_chunk("hdfc_ergo", "Room rent is capped at 1% of sum insured.", 12)],
        "Show the room rent rule for HDFC Ergo Optima Super as a table.",
        table=True,
    )

    assert "Answer with a GitHub-flavoured Markdown table only" in captured["prompt"]
    assert "| Aspect | What the wording says | Where |" not in captured["prompt"]
