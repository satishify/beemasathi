from agents.orchestrator import BeemaSathiOrchestrator
from agents.router import IntentRouter
from rag.catalog import entry_line, entry_vehicle


def test_motor_availability_routes_to_catalog_list():
    plan = IntentRouter().route("is there any motor insurance wording available?")
    assert plan.intent == "catalog_list"
    assert plan.line == "motor"


def test_compare_and_list_without_insurers_is_a_listing():
    plan = IntentRouter().route("compare and list down 5 motor insurance")
    assert plan.intent == "catalog_list"
    assert plan.limit == 5


def test_two_wheeler_listing_ignores_class_named_policy():
    catalog = [{"insurer": "bajaj_allianz", "policy_name": "two_wheeler"}]
    plan = IntentRouter(catalog).route("list 5 two wheeler policies")
    assert plan.intent == "catalog_list"
    assert plan.vehicle == "two_wheeler"
    assert plan.policies == []


def test_line_clause_question_stays_coverage():
    assert IntentRouter().route("does motor insurance cover flood damage?").intent == "coverage"
    assert IntentRouter().route("what is the waiting period in motor insurance?").intent == "coverage"


def test_two_named_insurers_still_compare():
    plan = IntentRouter().route("compare star health and hdfc ergo room rent")
    assert plan.intent == "comparison"


def test_catalog_line_and_vehicle_classification():
    assert entry_line({"policy_name": "private_car_package_policy"}) == "motor"
    assert entry_line({"insurer": "star_health", "policy_name": "comprehensive"}) == "health"
    assert entry_line({"policy_name": "my_health_suraksha"}) == "health"
    assert entry_vehicle({"policy_name": "long_term_two_wheeler_package"}) == "two_wheeler"
    assert entry_vehicle({"policy_name": "private_car_standalone_own_damage"}) == "car"


def _entry(insurer, name, chunks=10, doc="policy_wording"):
    return {
        "insurer": insurer,
        "policy_name": name,
        "document_type": doc,
        "source_file": f"{insurer}_{name}.pdf",
        "chunks": chunks,
    }


def test_listing_is_grounded_and_line_filtered():
    catalog = [
        _entry("digit", "digit_motor_secure_cover", 100),
        _entry("bajaj_allianz", "two_wheeler_package_policy", 50),
        _entry("hdfc_ergo", "my_health_suraksha", 10),
        _entry("digit", "digit_health_plus", 20),
    ]
    orchestrator = BeemaSathiOrchestrator.__new__(BeemaSathiOrchestrator)
    plan = IntentRouter(catalog).route("compare and list down 5 motor insurance")
    result = orchestrator._catalog_list(plan, catalog)
    answer = result["answer"]

    assert result["plan"]["intent"] == "catalog_list"
    assert "2 motor policy wordings" in answer
    assert "Digit Motor Secure Cover" in answer
    assert "Two Wheeler Package Policy" in answer
    assert "My Health Suraksha" not in answer
    assert "can't say which plan" in answer
    assert result["retrieved"] == 0


def test_listing_respects_requested_count():
    catalog = [
        _entry("bajaj_allianz", "two_wheeler_package_policy", 60),
        _entry("bajaj_allianz", "two_wheeler_standalone_own_damage", 50),
        _entry("digit", "digit_two_wheeler_insurance", 40),
        _entry("tata_aig", "long_term_two_wheeler_package_policy", 30),
    ]
    orchestrator = BeemaSathiOrchestrator.__new__(BeemaSathiOrchestrator)
    plan = IntentRouter(catalog).route("list 2 motor policies")
    answer = orchestrator._catalog_list(plan, catalog)["answer"]

    assert "4 motor policy wordings" in answer
    assert answer.count("\n- ") == 2
