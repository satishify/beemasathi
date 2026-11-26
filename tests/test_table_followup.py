"""Table follow-ups keep their shape and inherit the previous question."""

from agents.orchestrator import BeemaSathiOrchestrator
from rag.generator import BimaSaathiGenerator, TABLE_FORMAT_INSTRUCTION, table_requested


def _bot():
    bot = BimaSaathiGenerator.__new__(BimaSaathiGenerator)
    bot.model = "test-model"
    bot.use_api = False
    return bot


def test_table_request_is_detected():
    assert table_requested("give me in table format")
    assert table_requested("show it as a grid")
    assert not table_requested("what is the room rent cap?")


def test_sanitize_keeps_table_rows():
    answer = (
        "| Insurer | Item | Detail |\n"
        "| --- | --- | --- |\n"
        "| HDFC Ergo | Room rent | 2% of sum insured, max Rs 5000/day |\n"
        "| Niva Bupa | Room rent | 2% of sum insured, max Rs 5000/day |"
    )

    assert _bot()._sanitize_answer(answer) == answer


def test_sanitize_keeps_the_blank_line_before_a_list():
    answer = "Two plans apply.\n\n- HDFC Ergo: 2% of sum insured\n- Niva Bupa: 2% of sum insured"

    assert _bot()._sanitize_answer(answer) == answer


def test_table_instruction_reaches_the_model(monkeypatch):
    bot = _bot()
    captured = {}

    def fake_call(prompt, messages=None):
        captured["prompt"] = prompt
        return "| Insurer | Cap |\n| --- | --- |\n| HDFC Ergo | 2% |", {"backend": "test"}

    monkeypatch.setattr(bot, "_call_model", fake_call)
    chunks = [
        {
            "text": "Room rent is capped at 2% of sum insured.",
            "metadata": {
                "insurer": "hdfc_ergo",
                "policy_name": "health_suraksha",
                "section": "Room Rent",
                "document_type": "policy_wording",
                "page": 1,
            },
        }
    ]

    result = bot.generate(chunks, "give me in table format", catalog=[], table=True)

    assert TABLE_FORMAT_INSTRUCTION.strip() in captured["prompt"]
    assert result["answer"].startswith("| Insurer |")


def test_format_followup_inherits_the_previous_question():
    orchestrator = BeemaSathiOrchestrator.__new__(BeemaSathiOrchestrator)
    history = [
        {"role": "user", "content": "What is the room rent cap for HDFC Ergo?"},
        {"role": "assistant", "content": "2% of sum insured, capped at Rs 5000/day."},
        {"role": "user", "content": "give me in table format"},
    ]

    assert (
        orchestrator._resolve_followup_question("give me in table format", history)
        == "What is the room rent cap for HDFC Ergo?"
    )
    # A question that names coverage retrieves on its own words.
    assert (
        orchestrator._resolve_followup_question(
            "can you show the room rent cap in a table?", history
        )
        == "can you show the room rent cap in a table?"
    )


class _RetryGenerator:
    model = "test-model"

    def __init__(self, retry_answer):
        self.retry_answer = retry_answer

    def generate(self, *args, **kwargs):
        return {
            "answer": self.retry_answer,
            "safe": True,
            "sources": [],
            "model_metrics": {},
        }


def test_retry_note_only_cites_figures_in_the_shown_answer():
    orchestrator = BeemaSathiOrchestrator.__new__(BeemaSathiOrchestrator)
    orchestrator.generator = _RetryGenerator("The cover lasts 9 days.")
    chunks = [
        {
            "text": "Room rent is capped at 2% of sum insured.",
            "metadata": {"insurer": "hdfc_ergo"},
        }
    ]
    result = {"answer": "The cover lasts 7 days.", "safe": True}

    verification = orchestrator._verify(
        result, chunks, "How long is the cover?", [], "baseline", [], False
    )

    assert "7 days" in result["answer"]
    assert "9 days" not in result["answer"]
    assert verification["unsupported"] == ["7 days"]