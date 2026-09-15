from types import SimpleNamespace

import groq
import httpx
import llm
import pytest
from config import settings
from pydantic import SecretStr

MESSAGES = [{"role": "user", "content": "hi"}]


def _fake_groq_client(tokens: list[str], error: Exception | None = None):
    def create(**kwargs):
        if error:
            raise error
        if kwargs.get("stream"):
            deltas = [*tokens, None]
            return iter(
                SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=t))])
                for t in deltas
            )
        message = SimpleNamespace(content="".join(tokens))
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


@pytest.fixture
def groq_settings(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "groq")
    monkeypatch.setattr(settings, "groq_api_key", SecretStr("test-key"))


@pytest.fixture
def groq_provider(groq_settings, monkeypatch):
    monkeypatch.setattr(llm, "_groq_client", lambda: _fake_groq_client(["Hello", " world"]))


class TestOllamaProvider:
    def test_chat_returns_content(self, mock_ollama):
        assert llm.chat(MESSAGES) == "The login function checks credentials."

    def test_stream_yields_tokens(self, mock_ollama):
        assert "".join(llm.stream_chat(MESSAGES)) == "The login function checks credentials."

    def test_connection_failure_raises_llm_error(self, monkeypatch):
        def refuse(self, **kwargs):
            raise ConnectionError("connection refused")

        monkeypatch.setattr("ollama.Client.chat", refuse)
        with pytest.raises(llm.LLMError):
            llm.chat(MESSAGES)

    def test_mid_stream_failure_raises_llm_error(self, monkeypatch):
        def broken_stream(self, **kwargs):
            yield {"message": {"content": "partial"}}
            raise ConnectionError("dropped")

        monkeypatch.setattr("ollama.Client.chat", broken_stream)
        with pytest.raises(llm.LLMError):
            list(llm.stream_chat(MESSAGES))

    def test_uses_configured_host(self, monkeypatch):
        monkeypatch.setattr(settings, "ollama_base_url", "http://ollama:11434")
        client = llm._ollama_client()
        assert "ollama:11434" in str(client._client.base_url)

    def test_active_model(self):
        assert llm.active_model() == settings.ollama_model


class TestGroqProvider:
    def test_chat_returns_content(self, groq_provider):
        assert llm.chat(MESSAGES) == "Hello world"

    def test_stream_skips_empty_deltas(self, groq_provider):
        assert list(llm.stream_chat(MESSAGES)) == ["Hello", " world"]

    def test_missing_api_key_raises(self, monkeypatch):
        monkeypatch.setattr(settings, "llm_provider", "groq")
        monkeypatch.setattr(settings, "groq_api_key", None)
        with pytest.raises(llm.LLMError, match="GROQ_API_KEY"):
            llm.chat(MESSAGES)

    def test_api_error_raises_llm_error(self, groq_settings, monkeypatch):
        error = groq.APIConnectionError(request=httpx.Request("POST", "https://api.groq.com"))
        monkeypatch.setattr(llm, "_groq_client", lambda: _fake_groq_client([], error=error))
        with pytest.raises(llm.LLMError):
            llm.chat(MESSAGES)

    def test_active_model(self, groq_provider):
        assert llm.active_model() == settings.groq_model
