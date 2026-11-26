"""Off-topic refusal and cross-insurer comparison from a product family."""

from agents.orchestrator import BeemaSathiOrchestrator
from agents.router import IntentRouter

CATALOG = [
    {"insurer": "max_bupa", "policy_name": "health_plus", "document_type": "brochure"},
    {
        "insurer": "digit",
        "policy_name": "digit_health_plus_policy_revision",
        "document_type": "policy_wording",
    },
]

QUESTION = (
    "Do you know any health plus policy where 2 different company have "
    "different constraint?"
)


class _Store:
    def __init__(self, catalog):
        self.catalog = catalog

    def get_catalog(self):
        return self.catalog


class _Retriever:
    def __init__(self, catalog):
        self.store = _Store(catalog)
        self.calls = []
        self.queries = []

    def retrieve(self, query, insurer=None, policy_name=None, n_results=8):
        self.calls.append((insurer, policy_name))
        self.queries.append(query)
        return [
            {
                "text": "Coverage is limited to the sum insured in the schedule.",
                "metadata": {
                    "insurer": insurer,
                    "policy_name": policy_name or insurer,
                    "source": f"{insurer}_demo",
                    "source_file": f"{insurer}_demo.pdf",
                    "page": 1,
                    "document_type": "policy_wording",
                },
            }
        ]


class _Generator:
    model = "test-model"

    def __init__(self):
        self.calls = []
        self.compares = []

    def generate(self, chunks, question, catalog=None, strategy="baseline", history=None, table=False):
        self.calls.append((question, [c["metadata"]["insurer"] for c in chunks]))
        return {
            "answer": "Confirmed from the loaded wording.",
            "safe": True,
            "sources": [],
            "model_metrics": {},
        }

    def compare_plans(self, question, groups, history=None):
        self.compares.append(
            (
                question,
                [(label, [c["metadata"]["insurer"] for c in chunks]) for label, chunks in groups],
            )
        )
        return {
            "answer": "| Constraint | A | B |\n| --- | --- | --- |\n| Room rent | 1% | 2% |",
            "safe": True,
            "sources": [],
            "model_metrics": {},
        }


def _orchestrator(catalog=None):
    generator = _Generator()
    retriever = _Retriever(catalog or [])
    orchestrator = BeemaSathiOrchestrator(
        retriever=retriever, generator=generator, predictor=None
    )
    orchestrator.router = IntentRouter(catalog or [])
    return orchestrator, generator, retriever


def test_off_topic_question_never_reaches_retrieval_or_the_model():
    orchestrator, generator, retriever = _orchestrator()

    result = orchestrator.answer("what is capital of France?")

    assert "only answer questions about the Indian health and motor insurance" in result["answer"]
    assert generator.calls == []
    assert retriever.calls == []
    assert result["retrieved"] == 0


def test_insurance_question_without_common_keywords_still_passes():
    orchestrator, generator, _ = _orchestrator()

    orchestrator.answer(
        "If I need ICU care, what daily ICU charge limit does HDFC ERGO "
        "Arogya Sanjeevani state?"
    )

    assert generator.calls, "an ICU question must reach the model"


def test_product_family_comparison_retrieves_each_plan_once():
    orchestrator, generator, retriever = _orchestrator(CATALOG)

    result = orchestrator.answer(QUESTION)

    assert retriever.calls == [
        ("digit", "digit_health_plus_policy_revision"),
        ("max_bupa", "health_plus"),
    ]
    assert "Comparing the Health Plus plans I hold for" in result["answer"]
    assert len(generator.compares) == 1
    assert generator.calls == []
    _, groups = generator.compares[0]
    assert [insurers for _, insurers in groups] == [["digit"], ["max_bupa"]]
    assert "| Constraint |" in result["answer"]


def test_multi_part_question_retrieves_each_aspect_on_its_own():
    orchestrator, _, retriever = _orchestrator()

    chunks = orchestrator._retrieve_for_question(
        "What are the waiting period and co-pay for HDFC Ergo Optima Super?",
        "hdfc_ergo",
        "optima_super",
    )

    assert retriever.queries == [
        "What are the waiting period and co-pay for HDFC Ergo Optima Super?",
        "waiting period",
        "co-pay",
    ]
    assert len(chunks) == 3


def test_one_part_question_retrieves_once():
    orchestrator, _, retriever = _orchestrator()

    orchestrator._retrieve_for_question(
        "What is the room rent cap for HDFC Ergo Optima Super?",
        "hdfc_ergo",
        "optima_super",
    )

    assert len(retriever.queries) == 1


def test_comparison_with_nothing_to_compare_asks_for_a_second_plan():
    orchestrator, generator, _ = _orchestrator()

    result = orchestrator.answer(
        "Do you know any policy where 2 different companies differ?"
    )

    assert "I need two plans" in result["answer"]
    assert generator.calls == []


def test_vague_comparison_lists_families_held_across_insurers():
    orchestrator, generator, _ = _orchestrator(CATALOG)

    result = orchestrator.answer(
        "Do you know any policy where purpose is same but 2 different company "
        "have different constraint?"
    )

    assert "**Health Plus** — " in result["answer"]
    assert "Niva Bupa (Max)" in result["answer"]
    assert generator.calls == []
