import pytest

from rag import generator as gen


class _FailingHosted(gen.BimaSaathiGenerator):
    def __init__(self):
        self.use_api = True
        self.model = "Qwen/Qwen3-8B"
        self.local_models = []

    def _call_api(self, messages, max_tokens=None):
        raise RuntimeError("Model API error 503: busy")

    def _call_local(self, prompt, chat_messages, model=None):
        self.local_models.append(model)
        return "local answer", {"backend": "ollama-python", "model": model}


def test_hosted_failure_retires_to_local_ollama(monkeypatch):
    monkeypatch.setattr(gen, "_local_backend_ready", lambda: True)
    bot = _FailingHosted()

    answer, metrics = bot._call_model("question")

    assert answer == "local answer"
    assert bot.local_models == [gen.LLM_LOCAL]
    assert metrics["fallback"] == "hosted-unavailable"
    assert "503" in metrics["hosted_error"]


def test_hosted_error_surfaces_when_no_local_backend(monkeypatch):
    monkeypatch.setattr(gen, "_local_backend_ready", lambda: False)
    bot = _FailingHosted()

    with pytest.raises(RuntimeError, match="503"):
        bot._call_model("question")

    assert bot.local_models == []


def test_local_only_mode_never_calls_hosted(monkeypatch):
    bot = _FailingHosted()
    bot.use_api = False
    monkeypatch.setattr(
        gen, "_local_backend_ready", lambda: pytest.fail("should not probe")
    )

    answer, metrics = bot._call_model("question")

    assert answer == "local answer"
    assert bot.local_models == [None]
    assert "fallback" not in metrics


def test_relative_ca_path_resolves_against_project_root(monkeypatch):
    monkeypatch.setattr(gen, "LLM_API_CA", "certs/ca.crt")
    assert gen._hosted_ca_verify() == str(gen._ROOT_DIR / "certs" / "ca.crt")


def test_blank_ca_keeps_default_verification(monkeypatch):
    monkeypatch.setattr(gen, "LLM_API_CA", "")
    assert gen._hosted_ca_verify() is True

