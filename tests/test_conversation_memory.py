"""Conversation memory: recent turns reach the prompt, chat-meta is not blocked."""

from agents.orchestrator import BeemaSathiOrchestrator
from rag.generator import BimaSaathiGenerator


def _bot():
    bot = BimaSaathiGenerator.__new__(BimaSaathiGenerator)
    bot.model = "test-model"
    bot.use_api = False
    return bot


def test_history_text_labels_roles():
    history = [
        {"role": "user", "content": "can i only take own damage insurance ?"},
        {"role": "assistant", "content": "Yes, standalone own damage plans exist."},
    ]

    text = _bot()._history_text(history)

    assert text.startswith("Customer: can i only take own damage insurance ?")
    assert "BeemaSathi: Yes, standalone own damage plans exist." in text


def test_history_text_is_bounded():
    history = [{"role": "user", "content": "q" * 800} for _ in range(20)]

    text = _bot()._history_text(history)

    assert len(text) <= 2400
    assert text.count("Customer:") <= 8


def test_empty_history_is_stated_explicitly():
    assert "first question" in _bot()._history_text([])


def test_chat_meta_questions_are_detected():
    bot = _bot()

    assert bot._is_chat_meta_question("what was my first question in this chat context ?")
    assert bot._is_chat_meta_question("summarise our chat so far")
    assert not bot._is_chat_meta_question("does the policy cover own damage?")


def test_chat_meta_question_reaches_model_with_history(monkeypatch):
    bot = _bot()
    captured = {}

    def fake_call(prompt, messages=None):
        captured["prompt"] = prompt
        return "Your first question was about own damage cover.", {"backend": "test"}

    monkeypatch.setattr(bot, "_call_model", fake_call)

    result = bot.generate(
        [],
        "what was my first question in this chat context ?",
        catalog=[],
        history=[{"role": "user", "content": "can i only take own damage insurance ?"}],
    )

    assert "RECENT CONVERSATION" in captured["prompt"]
    assert "can i only take own damage insurance ?" in captured["prompt"]
    assert result["safe"] is True


def test_coverage_gap_still_refuses_without_history(monkeypatch):
    bot = _bot()
    monkeypatch.setattr(
        bot, "_call_model", lambda *a, **k: (_ for _ in ()).throw(AssertionError)
    )

    result = bot.generate([], "Does Acko cover maternity?", catalog=[])

    assert "I don't have that policy wording" in result["answer"]


class _Store:
    def get_catalog(self):
        return []


class _Retriever:
    def __init__(self):
        self.store = _Store()

    def retrieve(self, **kwargs):
        return []


class _Generator:
    model = "test-model"

    def __init__(self):
        self.calls = []

    def generate(self, chunks, question, catalog=None, strategy="baseline", history=None):
        self.calls.append(history)
        return {
            "answer": "ok",
            "safe": True,
            "sources": [],
            "model_metrics": {"backend": "test"},
        }


class _Predictor:
    pass


def test_orchestrator_forwards_history_to_generator():
    generator = _Generator()
    orchestrator = BeemaSathiOrchestrator(
        retriever=_Retriever(), generator=generator, predictor=_Predictor()
    )
    history = [{"role": "user", "content": "can i only take own damage insurance ?"}]

    orchestrator.answer("3rd party insurance can be bought at different time?", history=history)

    assert generator.calls == [history]
