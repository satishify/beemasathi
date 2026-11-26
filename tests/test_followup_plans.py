"""Follow-ups that point back at the plans the previous answer listed."""

from agents import orchestrator as orchestrator_module
from agents.orchestrator import BeemaSathiOrchestrator
from agents.router import IntentRouter

NIVA = [
    "reassure_2_0",
    "reassure_1_0",
    "health_companion",
    "senior_first",
    "health_premia",
]

CATALOG = [
    {
        "insurer": "niva_bupa",
        "policy_name": name,
        "document_type": "policy_wording",
        "source_file": f"{name}.pdf",
        "chunks": 100 - index,
    }
    for index, name in enumerate(NIVA)
] + [
    {
        "insurer": "star_health",
        "policy_name": "comprehensive",
        "document_type": "policy_wording",
        "source_file": "star.pdf",
        "chunks": 50,
    },
    {
        "insurer": "hdfc_ergo",
        "policy_name": "optima_super",
        "document_type": "policy_wording",
        "source_file": "hdfc.pdf",
        "chunks": 50,
    },
]

FOCUS = [{"insurer": "niva_bupa", "policy": name} for name in NIVA]


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
        self.calls.append((insurer, policy_name, n_results))
        self.queries.append(query)
        return [
            {
                "text": "The room rent cap is 1% of sum insured per day.",
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
        self.calls.append(question)
        return {
            "answer": "Confirmed from the loaded wording.",
            "safe": True,
            "sources": [],
            "model_metrics": {},
        }

    def compare_plans(self, question, groups, history=None):
        self.compares.append((question, [label for label, _ in groups]))
        header = "| Constraint | " + " | ".join(label for label, _ in groups) + " |"
        sep = "| --- |" + " --- |" * len(groups)
        return {
            "answer": header + "\n" + sep + "\n| Room rent | 1% |",
            "safe": True,
            "sources": [],
            "model_metrics": {},
        }


def _orchestrator(catalog=None):
    generator = _Generator()
    retriever = _Retriever(catalog if catalog is not None else CATALOG)
    instance = BeemaSathiOrchestrator(
        retriever=retriever, generator=generator, predictor=None
    )
    instance.router = IntentRouter(catalog if catalog is not None else CATALOG)
    return instance, generator, retriever


def test_catalog_list_hands_back_raw_policy_names():
    instance, _, _ = _orchestrator()
    plan = IntentRouter(CATALOG).route("list niva bupa health plans")
    result = instance._catalog_list(plan, CATALOG)

    listed = [item["policy"] for item in result["focus_plans"]]
    assert listed == NIVA
    assert all(item["insurer"] == "niva_bupa" for item in result["focus_plans"])


def test_compare_each_after_a_listing_compares_every_listed_plan():
    instance, generator, retriever = _orchestrator()

    result = instance.answer(
        "can you compare each ?",
        followup_plans=FOCUS,
        history=[{"role": "assistant", "content": "Here are 5:"}],
    )

    assert orchestrator_module.OFF_TOPIC_ANSWER != result["answer"]
    assert len(generator.compares) == 1
    question, labels = generator.compares[0]
    assert question == orchestrator_module.FOCUS_ASPECTS_DIFFERENCES
    assert len(labels) == orchestrator_module.FOCUS_PLAN_CAP
    assert set(call[1] for call in retriever.calls) == set(NIVA[:4])
    assert all(call[0] == "niva_bupa" for call in retriever.calls)
    assert set(retriever.queries) == {orchestrator_module.FOCUS_RETRIEVAL_QUERY}
    assert "I compared 4 of those 5 plans." in result["answer"]


def test_each_retrieval_stays_within_the_context_budget():
    instance, _, retriever = _orchestrator()

    instance.answer("compare each", followup_plans=FOCUS)

    assert sum(call[2] for call in retriever.calls) <= 12


def test_misspelled_insurer_follow_up_uses_the_listed_plans():
    instance, generator, _ = _orchestrator()

    instance.answer(
        "can you compare each of above niva bua health plan ?",
        followup_plans=FOCUS,
    )

    assert len(generator.compares) == 1
    _, labels = generator.compares[0]
    assert len(labels) == orchestrator_module.FOCUS_PLAN_CAP


def test_three_listed_plans_compare_on_the_named_aspects():
    instance, generator, _ = _orchestrator()

    instance.answer("compare all of them", followup_plans=FOCUS[:3])

    question, labels = generator.compares[0]
    assert question == orchestrator_module.FOCUS_ASPECTS
    assert len(labels) == 3


def test_compare_each_without_a_listing_is_still_refused():
    instance, generator, retriever = _orchestrator()

    result = instance.answer("can you compare each ?")

    assert result["answer"] == orchestrator_module.OFF_TOPIC_ANSWER
    assert generator.compares == []
    assert retriever.calls == []


def test_named_comparison_is_not_hijacked_by_the_previous_listing():
    instance, generator, retriever = _orchestrator()

    instance.answer(
        "compare star health and hdfc ergo room rent",
        followup_plans=FOCUS,
    )

    _, labels = generator.compares[0]
    assert len(labels) == 2
    assert {call[0] for call in retriever.calls} == {"star_health", "hdfc_ergo"}


def test_follow_up_plans_of_one_plan_ask_for_a_second_plan():
    instance, generator, _ = _orchestrator()

    result = instance.answer(
        "can you compare each ?", followup_plans=FOCUS[:1]
    )

    assert generator.compares == []
    assert "Name a second plan or insurer" in result["answer"]
