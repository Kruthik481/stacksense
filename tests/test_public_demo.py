import llm
import pytest
from config import settings

ASK = {"query": "how does login work?"}


@pytest.fixture
def demo_client(client, monkeypatch):
    from main import llm_rate_limiter

    monkeypatch.setattr(settings, "public_demo", True)
    llm_rate_limiter.reset()
    yield client
    llm_rate_limiter.reset()


class TestPublicDemoLocksWrites:
    @pytest.mark.parametrize(
        ("method", "path", "body"),
        [
            ("POST", "/api/ingest", {"path": "/etc"}),
            ("POST", "/api/ingest/start", {"path": "/etc"}),
            ("POST", "/api/projects", {"name": "x", "path": "/etc"}),
            ("DELETE", "/api/projects/abc", None),
        ],
    )
    def test_returns_403(self, demo_client, method, path, body):
        r = demo_client.request(method, path, json=body)
        assert r.status_code == 403

    def test_health_reports_demo_mode(self, demo_client):
        data = demo_client.get("/api/health").json()
        assert data["public_demo"] is True
        assert data["provider"] == settings.llm_provider


class TestQueryValidation:
    def test_empty_query_rejected(self, client):
        assert client.post("/api/ask", json={"query": ""}).status_code == 422

    def test_oversized_query_rejected(self, client):
        from main import MAX_QUERY_CHARS

        r = client.post("/api/ask", json={"query": "x" * (MAX_QUERY_CHARS + 1)})
        assert r.status_code == 422


class TestRateLimit:
    def test_llm_endpoints_limited_in_demo(self, demo_client, monkeypatch):
        monkeypatch.setattr(settings, "rate_limit_per_minute", 2)
        statuses = [demo_client.post("/api/ask", json=ASK).status_code for _ in range(3)]
        assert statuses == [200, 200, 429]

    def test_stream_shares_the_limit(self, demo_client, monkeypatch):
        monkeypatch.setattr(settings, "rate_limit_per_minute", 1)
        assert demo_client.post("/api/ask", json=ASK).status_code == 200
        assert demo_client.post("/api/stream", json=ASK).status_code == 429

    def test_not_limited_outside_demo(self, client, monkeypatch):
        monkeypatch.setattr(settings, "rate_limit_per_minute", 1)
        statuses = [client.post("/api/ask", json=ASK).status_code for _ in range(2)]
        assert statuses == [200, 200]


class TestLLMFailure:
    def test_ask_returns_503(self, client, monkeypatch):
        def fail(messages):
            raise llm.LLMError("The language model is unavailable right now.")

        monkeypatch.setattr(llm, "chat", fail)
        r = client.post("/api/ask", json=ASK)
        assert r.status_code == 503
        assert "unavailable" in r.json()["detail"]

    def test_stream_emits_error_event(self, client, monkeypatch):
        def fail(messages):
            yield "partial"
            raise llm.LLMError("The language model is unavailable right now.")

        monkeypatch.setattr(llm, "stream_chat", fail)
        body = client.post("/api/stream", json=ASK).text
        assert '"type": "error"' in body
        assert '"type": "done"' not in body

    def test_failed_stream_does_not_save_messages(self, client, monkeypatch):
        def fail(messages):
            raise llm.LLMError("down")
            yield  # pragma: no cover - makes this a generator

        monkeypatch.setattr(llm, "stream_chat", fail)
        session = client.post("/api/sessions", json={"title": "Q"}).json()
        client.post("/api/stream", json={**ASK, "session_id": session["id"]})
        assert client.get(f"/api/sessions/{session['id']}/messages").json() == []
