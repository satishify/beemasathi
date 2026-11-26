"""Structured comparison: one call, both plans labelled, differences-only table."""

from rag.generator import BimaSaathiGenerator


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
            "policy_name": f"{insurer}_health_plus",
            "source": f"{insurer}_demo",
            "source_file": f"{insurer}_demo.pdf",
            "section": f"Sec. {page}",
            "page": page,
            "document_type": "policy_wording",
        },
    }


GROUPS = [
    ("Alpha Health Plus", [_chunk("alpha", "Room rent is 1% of sum insured per day.", 4)]),
    ("Beta Health Plus", [_chunk("beta", "Room rent is 2% of sum insured per day.", 9)]),
]

TABLE = (
    "| Constraint | Alpha Health Plus | Beta Health Plus |\n"
    "| --- | --- | --- |\n"
    "| Room rent | 1% of sum insured per day (Sec. 4, p. 4) | "
    "2% of sum insured per day (Sec. 9, p. 9) |"
)


def test_compare_plans_makes_one_labelled_call(monkeypatch):
    bot = _bot()
    captured = {}

    def fake_call(prompt, messages=None, max_tokens=None):
        captured["prompt"] = prompt
        captured["messages"] = messages
        return TABLE, {"backend": "test"}

    monkeypatch.setattr(bot, "_call_model", fake_call)

    result = bot.compare_plans("Compare room rent for both plans.", GROUPS)

    assert captured["messages"][0]["role"] == "system"
    assert "=== Alpha Health Plus ===" in captured["prompt"]
    assert "=== Beta Health Plus ===" in captured["prompt"]
    assert "STEP 3 - ALIGN" in captured["prompt"]
    assert result["answer"].startswith(TABLE)
    assert "This is informational only." in result["answer"]
    assert result["safe"] is True


def test_prose_before_the_table_is_dropped(monkeypatch):
    bot = _bot()
    text = "Alpha Health Plus vs Beta Health Plus on room rent.\n\n" + TABLE
    monkeypatch.setattr(bot, "_call_model", lambda *a, **k: (text, {"backend": "test"}))

    result = bot.compare_plans("Compare room rent for both plans.", GROUPS)

    assert result["answer"].startswith("| Constraint | Alpha Health Plus")
    assert "on room rent." not in result["answer"]


def test_compare_plans_restores_a_dropped_header(monkeypatch):
    bot = _bot()
    mangled = TABLE.split("\n", 1)[1].replace("| --- | --- | --- |", "| --- | --- |")
    mangled = "Alpha Health Plus | Beta Health Plus\n" + mangled
    monkeypatch.setattr(bot, "_call_model", lambda *a, **k: (mangled, {"backend": "test"}))

    result = bot.compare_plans("Compare room rent for both plans.", GROUPS)

    assert result["answer"].startswith(
        "| Constraint | Alpha Health Plus | Beta Health Plus |"
    )
    assert "| --- | --- | --- |" in result["answer"].split("\n")[1]


def test_differences_only_rows_and_three_bullets(monkeypatch):
    bot = _bot()
    text = (
        TABLE
        + "\n| Sub-limits | Not stated | Not stated |"
        + "\n| Renewal | Not stated | ReAssure benefit (Sec."
        + "\n\n- one\n- two\n- three\n- four\n\nThis is informational only."
    )
    monkeypatch.setattr(bot, "_call_model", lambda *a, **k: (text, {"backend": "test"}))

    result = bot.compare_plans("Compare room rent for both plans.", GROUPS)

    assert "Sub-limits" not in result["answer"]
    assert "Renewal" not in result["answer"]
    assert "- three" in result["answer"]
    assert "- four" not in result["answer"]
    assert "This is informational only." in result["answer"]


def test_identical_rows_collapse_to_a_no_difference_answer(monkeypatch):
    bot = _bot()
    text = (
        "| Constraint | Alpha Health Plus | Beta Health Plus |\n"
        "| --- | --- | --- |\n"
        "| Room rent | 1% of sum insured (Sec. 4, p. 4) | 1% of sum insured (Sec. 9, p. 9) |\n\n"
        "- Both plans are the same."
    )
    monkeypatch.setattr(bot, "_call_model", lambda *a, **k: (text, {"backend": "test"}))

    result = bot.compare_plans("What constraints differ between these plans?", GROUPS)

    assert "| Constraint |" not in result["answer"]
    assert "no constraint where they state different rules" in result["answer"]
    assert "Both plans are the same" not in result["answer"]


def test_side_by_side_keeps_matching_rules(monkeypatch):
    bot = _bot()
    text = (
        "| Constraint | Alpha Health Plus | Beta Health Plus |\n"
        "| --- | --- | --- |\n"
        "| Room rent | 1% of sum insured (Sec. 4, p. 4) | 1% of sum insured (Sec. 9, p. 9) |\n\n"
        "- Both plans cap room rent at 1% of sum insured."
    )
    monkeypatch.setattr(bot, "_call_model", lambda *a, **k: (text, {"backend": "test"}))

    result = bot.compare_plans("Compare the room rent limit for both plans.", GROUPS)

    assert "| Room rent |" in result["answer"]
    assert "Both plans cap room rent" in result["answer"]


def test_differences_question_gets_the_differentiate_step(monkeypatch):
    bot = _bot()
    captured = {}

    def fake_call(prompt, messages=None, max_tokens=None):
        captured["prompt"] = prompt
        return TABLE, {"backend": "test"}

    monkeypatch.setattr(bot, "_call_model", fake_call)

    bot.compare_plans("What constraints differ between these plans?", GROUPS)

    assert "STEP 3 - DIFFERENTIATE" in captured["prompt"]
    assert "Never write that the plans are the same" in captured["prompt"]


def test_plan_name_in_question_does_not_drop_the_other_plan(monkeypatch):
    bot = _bot()
    captured = {}

    def fake_call(prompt, messages=None, max_tokens=None):
        captured["prompt"] = prompt
        return TABLE, {"backend": "test"}

    monkeypatch.setattr(bot, "_call_model", fake_call)
    groups = [
        ("Alpha Health Plus", [_chunk("alpha", "Room rent is 1% of sum insured per day.", 4)]),
        (
            "Beta Arogya Sanjeevani",
            [_chunk("beta", "The Arogya Sanjeevani room rent is 2% of sum insured per day.", 9)],
        ),
    ]

    result = bot.compare_plans(
        "Compare the room rent limit for Alpha Health Plus and Beta Arogya Sanjeevani.",
        groups,
    )

    assert "=== Alpha Health Plus ===" in captured["prompt"]
    assert "I don't have that policy wording" not in result["answer"]
    assert result["safe"] is True


def test_side_by_side_with_no_stated_rules_drops_the_empty_table(monkeypatch):
    bot = _bot()
    text = (
        "| Constraint | Alpha Health Plus | Beta Health Plus |\n"
        "| --- | --- | --- |\n"
        "| Room rent | Not stated | Not stated |\n"
        "| Co-pay | Not stated | Not stated |\n\n"
        "- Both plans do not specify room rent or co-pay."
    )
    monkeypatch.setattr(bot, "_call_model", lambda *a, **k: (text, {"backend": "test"}))

    result = bot.compare_plans("Compare the room rent limit for both plans.", GROUPS)

    assert "| Constraint |" not in result["answer"]
    assert "does not state a rule for what you asked" in result["answer"]
    assert "informational only" in result["answer"].lower()


def test_compare_request_stays_within_the_provider_token_cap(monkeypatch):
    bot = _bot()
    captured = {}

    def fake_call(prompt, messages=None, max_tokens=None):
        captured["max_tokens"] = max_tokens
        return TABLE, {"backend": "test"}

    monkeypatch.setattr(bot, "_call_model", fake_call)

    bot.compare_plans("Compare room rent for both plans.", GROUPS)

    assert captured["max_tokens"] <= 1024


def test_ragged_rows_are_dropped_because_they_cannot_be_attributed(monkeypatch):
    bot = _bot()
    text = (
        TABLE
        + "\n| Co-pay | 10% co-pay on each claim (Sec. 6, p. 7) |"
        + "\n| Waiting period | 30 days (Sec. 2, p. 3) | 60 days (Sec. 5, p. 6) |"
    )
    monkeypatch.setattr(bot, "_call_model", lambda *a, **k: (text, {"backend": "test"}))

    result = bot.compare_plans("Compare room rent for both plans.", GROUPS)

    assert "Room rent" in result["answer"]
    assert "Co-pay" not in result["answer"]
    assert "Waiting period" in result["answer"]


def test_wide_tables_get_a_cell_width_note(monkeypatch):
    bot = _bot()
    captured = {}

    def fake_call(prompt, messages=None, max_tokens=None):
        captured["prompt"] = prompt
        return TABLE, {"backend": "test"}

    monkeypatch.setattr(bot, "_call_model", fake_call)

    bot.compare_plans("Compare room rent for all three plans.", GROUPS + GROUPS[:1])

    assert "WIDTH: this table has 3 plan columns" in captured["prompt"]


def test_two_plan_tables_do_not_get_the_width_note(monkeypatch):
    bot = _bot()
    captured = {}

    def fake_call(prompt, messages=None, max_tokens=None):
        captured["prompt"] = prompt
        return TABLE, {"backend": "test"}

    monkeypatch.setattr(bot, "_call_model", fake_call)

    bot.compare_plans("Compare room rent for both plans.", GROUPS)

    assert "WIDTH:" not in captured["prompt"]


def test_compare_plans_needs_two_groups(monkeypatch):
    bot = _bot()
    monkeypatch.setattr(
        bot, "_call_model", lambda *a, **k: (_ for _ in ()).throw(AssertionError)
    )

    result = bot.compare_plans("Compare room rent.", GROUPS[:1])

    assert result["safe"] is True
    assert "I don't have that policy wording" in result["answer"]
