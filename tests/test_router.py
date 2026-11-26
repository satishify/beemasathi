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


def test_same_insurer_comparison_resolves_both_plans():
    catalog = [
        {"insurer": "hdfc_ergo", "policy_name": "optima_super", "chunks": 5},
        {"insurer": "hdfc_ergo", "policy_name": "optima_vital", "chunks": 3},
        {"insurer": "hdfc_ergo", "policy_name": "optima_senior", "chunks": 2},
    ]
    router = IntentRouter(catalog)

    plan = router.route("Compare HDFC Ergo Optima Super and Optima Vital on room rent.")

    assert plan.intent == "comparison"
    assert plan.insurers == ["hdfc_ergo"]
    assert plan.topic_plans == {"hdfc_ergo": ["optima_super", "optima_vital"]}


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


FAMILY_CATALOG = [
    {
        "insurer": "max_bupa",
        "policy_name": "health_plus",
        "document_type": "brochure",
    },
    {
        "insurer": "digit",
        "policy_name": "digit_health_plus_policy_revision",
        "document_type": "policy_wording",
    },
    {
        "insurer": "aditya_birla",
        "policy_name": "arogya_sanjeevani",
        "document_type": "policy_wording",
    },
    {
        "insurer": "bajaj_allianz",
        "policy_name": "arogya_sanjeevani",
        "document_type": "policy_wording",
    },
]


def test_cross_insurer_comparison_resolves_the_named_product_family():
    router = IntentRouter(FAMILY_CATALOG)
    plan = router.route(
        "Do you know any health plus policy where 2 different company have "
        "different constraint?"
    )

    assert plan.intent == "comparison"
    assert plan.insurers == ["digit", "max_bupa"]
    assert plan.topic_plans["digit"] == ["digit_health_plus_policy_revision"]
    assert plan.lead_topic == "health plus"


def test_vague_comparison_offers_real_families_instead_of_guessing():
    router = IntentRouter(FAMILY_CATALOG)
    plan = router.route(
        "Do you know any policy where purpose is same but 2 different company "
        "have different constraint?"
    )

    assert plan.intent == "comparison"
    assert plan.insurers == []
    families = router.topic_families()
    assert families[0]["label"] == "arogya sanjeevani"
    assert sorted(families[0]["plans"]) == ["aditya_birla", "bajaj_allianz"]


def test_cross_insurer_question_without_any_match_stays_a_comparison():
    router = IntentRouter()
    plan = router.route("Do you know any policy where 2 different companies differ?")
    assert plan.intent == "comparison"
    assert plan.insurers == []


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
