class TestHealthEndpoint:
    def test_returns_ok(self, client):
        r = client.get("/api/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        assert data["index_loaded"] is True
        assert data["vectors"] == 3

    def test_includes_model_info(self, client):
        r = client.get("/api/health")
        data = r.json()
        assert "model" in data
        assert "embedding_model" in data


class TestSessionsAPI:
    def test_create_session(self, client):
        r = client.post("/api/sessions", json={"title": "Test Chat"})
        assert r.status_code == 200
        data = r.json()
        assert data["title"] == "Test Chat"
        assert "id" in data

    def test_list_sessions(self, client):
        client.post("/api/sessions", json={"title": "A"})
        client.post("/api/sessions", json={"title": "B"})
        r = client.get("/api/sessions")
        assert r.status_code == 200
        assert len(r.json()) >= 2

    def test_get_session(self, client):
        created = client.post("/api/sessions", json={"title": "Find Me"}).json()
        r = client.get(f"/api/sessions/{created['id']}")
        assert r.status_code == 200
        assert r.json()["title"] == "Find Me"

    def test_get_nonexistent_session(self, client):
        r = client.get("/api/sessions/does-not-exist")
        assert r.status_code == 404

    def test_delete_session(self, client):
        created = client.post("/api/sessions", json={"title": "Delete Me"}).json()
        r = client.delete(f"/api/sessions/{created['id']}")
        assert r.status_code == 200
        r = client.get(f"/api/sessions/{created['id']}")
        assert r.status_code == 404

    def test_update_session(self, client):
        created = client.post("/api/sessions", json={"title": "Old"}).json()
        r = client.patch(f"/api/sessions/{created['id']}", json={"title": "New"})
        assert r.status_code == 200
        fetched = client.get(f"/api/sessions/{created['id']}").json()
        assert fetched["title"] == "New"


class TestMessagesAPI:
    def test_empty_messages(self, client):
        s = client.post("/api/sessions", json={"title": "Empty"}).json()
        r = client.get(f"/api/sessions/{s['id']}/messages")
        assert r.status_code == 200
        assert r.json() == []

    def test_messages_after_ask(self, client):
        s = client.post("/api/sessions", json={"title": "Chat"}).json()
        client.post("/api/ask", json={"query": "what is login?", "session_id": s["id"]})
        r = client.get(f"/api/sessions/{s['id']}/messages")
        msgs = r.json()
        assert len(msgs) == 2
        assert msgs[0]["role"] == "user"
        assert msgs[1]["role"] == "assistant"


class TestAskEndpoint:
    def test_returns_answer(self, client):
        r = client.post("/api/ask", json={"query": "how does login work?"})
        assert r.status_code == 200
        assert "answer" in r.json()
        assert len(r.json()["answer"]) > 0

    def test_with_session(self, client):
        s = client.post("/api/sessions", json={"title": "Q"}).json()
        r = client.post(
            "/api/ask",
            json={"query": "explain auth", "session_id": s["id"]},
        )
        assert r.status_code == 200

    def test_no_index_returns_503(self, client):
        from main import store

        orig_index, orig_meta = store.index, store.metadata
        store.index, store.metadata = None, None
        r = client.post("/api/ask", json={"query": "test"})
        assert r.status_code == 503
        store.index, store.metadata = orig_index, orig_meta


class TestStreamEndpoint:
    def test_returns_sse(self, client):
        r = client.post(
            "/api/stream",
            json={"query": "how does login work?"},
        )
        assert r.status_code == 200
        assert "text/event-stream" in r.headers["content-type"]

        body = r.text
        assert "sources" in body
        assert "token" in body
        assert "done" in body


class TestIngestEndpoint:
    def test_ingest_sample_repo(self, client, monkeypatch):
        import os

        sample = os.path.join(os.path.dirname(__file__), "..", "backend", "sample_repo")
        r = client.post("/api/ingest", json={"path": sample})
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        assert data["files_processed"] > 0
        assert data["chunks_created"] > 0

    def test_ingest_bad_path(self, client):
        r = client.post("/api/ingest", json={"path": "/nonexistent/path"})
        assert r.status_code == 400


class TestFrontendServed:
    def test_root_serves_html(self, client):
        r = client.get("/")
        assert r.status_code == 200
        assert "StackSense" in r.text
