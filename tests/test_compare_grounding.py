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

    def generate(self, chunks, question, catalog=None, strategy="baseline"):
        return {
            "answer": "Room rent is limited to a single standard AC room.",
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
    result = orchestrator._compare(
        QUESTION, plan, ["star_health", "hdfc_ergo"], "baseline", CATALOG
    )

    assert orchestrator.retriever.calls == [("hdfc_ergo", "optima_super")]
    assert "don't have the policy named for Star Health" in result["answer"]
    assert result["retrieved"] == 1
