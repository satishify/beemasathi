from agents.router import IntentRouter
from rag import retriever as retriever_module


def test_detects_insurers_and_comparison():
    router = IntentRouter()
    plan = router.route("Compare Star Health and Care Health room rent caps")
    assert plan.intent == "comparison"
    assert plan.insurers == ["star_health", "care_health"]


def test_scope_and_safety_intents():
    router = IntentRouter()
    assert router.route("Which insurers do you not have policy wording for?").intent == "scope"
    assert router.route("Can you help me backdate my policy?").intent == "safety"


def test_claim_intent():
    router = IntentRouter()
    plan = router.route("Will my claim be rejected if the hospital is outside the network?")
    assert plan.intent == "claim_risk"


def test_policy_detection_ignores_inserted_words():
    catalog = [
        {"insurer": "digit", "policy_name": "digit_complete_care_policy"},
        {"insurer": "digit", "policy_name": "digit_group_complete_protect_policy"},
    ]
    router = IntentRouter(catalog)
    plan = router.route("In Digit Health Complete Care Policy, what is the waiting period?")
    assert "digit_complete_care_policy" in plan.policies


def test_explicit_insurer_wins():
    router = IntentRouter()
    plan = router.route("What is the room rent cap?", insurer="star_health")
    assert plan.insurers == ["star_health"]
    assert plan.intent == "coverage"


def test_recommendation_intent():
    router = IntentRouter()
    plan = router.route("Suggest a good health insurance plan for my family")
    assert plan.intent == "recommend"


def test_named_plan_phrase_follows_the_insurer_mention():
    router = IntentRouter()
    question = (
        "Compare room rent for HDFC Ergo Optima Secure and Niva Bupa ReAssure 2.0?"
    )
    assert router.named_plan_phrase(question, "max_bupa") == "ReAssure 2.0"
    assert router.named_plan_phrase(question, "hdfc_ergo") == "Optima Secure"


def test_named_plan_phrase_is_empty_without_a_plan():
    router = IntentRouter()
    assert router.named_plan_phrase("Compare Niva Bupa and HDFC Ergo.", "max_bupa") == ""


def test_catalog_only_insurer_is_recognised_without_code_changes():
    catalog = [{"insurer": "future_general", "policy_name": "shield_five"}]
    router = IntentRouter(catalog)

    plan = router.route(
        "Compare Future General Shield Five with HDFC Ergo Optima Super"
    )

    assert plan.insurers == ["hdfc_ergo", "future_general"]
    assert plan.intent == "comparison"


class _FakeIndex:
    policies = [
        ("digit", "arogya_sanjeevani_policy_go_digit"),
        ("digit", "arogya_sanjeevani_policy_go_digit_arogaya"),
        ("digit", "digit_health_plus_policy_revision"),
        ("digit", "digit_supreme_care_policy"),
        ("bajaj_allianz", "arogya_sanjeevani_policy"),
    ]


def test_product_word_maps_digit_typo_to_policy(monkeypatch):
    monkeypatch.setattr(retriever_module, "get_index", lambda: _FakeIndex())
    retriever = retriever_module.PolicyRetriever.__new__(
        retriever_module.PolicyRetriever
    )
    name = retriever._detect_policy(
        "what are the pros and cons of digital health arogya", "digit"
    )
    assert name == "arogya_sanjeevani_policy_go_digit"
