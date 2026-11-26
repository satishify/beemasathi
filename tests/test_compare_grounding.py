"""Comparison answers stay on the plan each insurer was asked about."""

from agents.orchestrator import BeemaSathiOrchestrator
from agents.router import IntentRouter

QUESTION = (
    "Compare the room rent rules for HDFC Ergo Optima Super and "
    "Star Health Family Health Optima."
)

CATALOG = [
    {
        "insurer": "hdfc_ergo",
        "policy_name": "optima_super",
        "document_type": "prospectus",
        "source_file": "health__optima_super__prospectus.pdf",
        "chunks": 40,
    },
    {
        "insurer": "star_health",
        "policy_name": "super_health",
        "document_type": "policy_wording",
        "source_file": "health__super_health__policy_wording.pdf",
        "chunks": 40,
    },
]


class _Retriever:
    def __init__(self):
        self.calls = []

    def retrieve(self, query, insurer=None, policy_name=None, n_results=8):
        self.calls.append((insurer, policy_name))
        return [
            {
                "text": "Room rent is limited to a single standard AC room.",
                "metadata": {
                    "insurer": insurer,
                    "policy_name": policy_name or insurer,
                    "source": f"{insurer}_demo",
                    "source_file": f"{insurer}_demo.pdf",
                    "page": 3,
                    "document_type": "policy_wording",
                },
            }
        ]


class _Generator:
    model = "test-model"

    def __init__(self):
        self.compares = []

    def generate(self, chunks, question, catalog=None, strategy="baseline", history=None, table=False):
        return {
            "answer": "Room rent is limited to a single standard AC room.",
            "safe": True,
            "sources": [],
            "model_metrics": {},
        }

    def compare_plans(self, question, groups, history=None):
        self.compares.append([label for label, _ in groups])
        return {
            "answer": "| Constraint | A | B |\n| --- | --- | --- |\n| Room rent | AC | AC |",
            "safe": True,
            "sources": [],
            "model_metrics": {},
        }


def _orchestrator():
    orchestrator = BeemaSathiOrchestrator.__new__(BeemaSathiOrchestrator)
    orchestrator.retriever = _Retriever()
    orchestrator.generator = _Generator()
    orchestrator.router = IntentRouter(CATALOG)
    return orchestrator


def test_comparison_retrieves_only_the_named_plan_per_insurer():
    orchestrator = _orchestrator()
    plan = orchestrator.router.route(QUESTION)
    result = orchestrator._compare(QUESTION, plan, ["star_health", "hdfc_ergo"], CATALOG)

    assert orchestrator.retriever.calls == [("hdfc_ergo", "optima_super")]
    assert "I don't have the Family Health Optima wording" in result["answer"]
    assert "don't have the policy named" not in result["answer"]
    assert result["retrieved"] == 1


def test_missing_plan_names_the_plan_and_the_plans_held():
    orchestrator = _orchestrator()
    catalog = CATALOG + [
        {
            "insurer": "max_bupa",
            "policy_name": "health_companion",
            "document_type": "policy_wording",
            "source_file": "health__health_companion__policy_wording.pdf",
            "chunks": 90,
        },
        {
            "insurer": "max_bupa",
            "policy_name": "health_pulse",
            "document_type": "policy_wording",
            "source_file": "health__health_pulse__policy_wording.pdf",
            "chunks": 70,
        },
    ]
    question = (
        "Compare the room rent rules for HDFC Ergo Optima Super and "
        "Niva Bupa ReAssure 2.0."
    )
    plan = orchestrator.router.route(question)

    result = orchestrator._compare(question, plan, ["hdfc_ergo", "max_bupa"], catalog)

    assert "I don't have the ReAssure 2.0 wording" in result["answer"]
    assert "What I do hold for Niva Bupa (Max) includes Health Companion and Health Pulse." in result["answer"]
    assert "don't have the policy named" not in result["answer"]


def test_catalog_only_insurer_gets_the_same_missing_plan_note():
    catalog = CATALOG + [
        {
            "insurer": "future_general",
            "policy_name": "shield_five",
            "document_type": "policy_wording",
            "source_file": "health__shield_five__policy_wording.pdf",
            "chunks": 30,
        }
    ]
    orchestrator = _orchestrator()
    orchestrator.router = IntentRouter(catalog)
    question = (
        "Compare the room rent cap for HDFC Ergo Optima Super and "
        "Future General Shield 9."
    )
    plan = orchestrator.router.route(question)

    result = orchestrator._compare(question, plan, ["hdfc_ergo", "future_general"], catalog)

    assert "**Future General**" in result["answer"]
    assert "I don't have the Shield 9 wording" in result["answer"]
    assert "What I do hold for Future General includes Shield Five." in result["answer"]


def test_both_named_plans_reach_one_structured_call():
    orchestrator = _orchestrator()
    question = (
        "Compare the room rent rules for HDFC Ergo Optima Super and "
        "Star Health Super Health."
    )
    plan = orchestrator.router.route(question)

    result = orchestrator._compare(question, plan, ["hdfc_ergo", "star_health"], CATALOG)

    assert orchestrator.retriever.calls == [
        ("hdfc_ergo", "optima_super"),
        ("star_health", "super_health"),
    ]
    assert orchestrator.generator.compares == [
        ["HDFC Ergo Optima Super", "Star Health Super Health"]
    ]
    assert result["answer"].startswith("| Constraint |")
    assert result["retrieved"] == 2


def test_absent_insurer_says_the_wording_is_not_loaded():
    orchestrator = _orchestrator()
    question = (
        "Compare the room rent rules for HDFC Ergo Optima Super and "
        "Niva Bupa ReAssure 2.0."
    )
    plan = orchestrator.router.route(question)

    result = orchestrator._compare(question, plan, ["hdfc_ergo", "max_bupa"], CATALOG)

    assert "I don't have a Niva Bupa (Max) policy wording loaded" in result["answer"]
    assert "ReAssure 2.0" not in result["answer"]
