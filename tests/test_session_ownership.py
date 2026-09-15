import sqlite3

import pytest
from config import settings
from sessions import create_session, init_db, list_sessions

ALICE = {"X-Client-Id": "alice-0000-0000-0000"}
BOB = {"X-Client-Id": "bob-0000-0000-0000-0"}
ASK = {"query": "how does login work?"}


@pytest.fixture
def demo_client(client, monkeypatch):
    from main import llm_rate_limiter

    monkeypatch.setattr(settings, "public_demo", True)
    llm_rate_limiter.reset()
    yield client
    llm_rate_limiter.reset()


@pytest.fixture
def alice_session(demo_client):
    return demo_client.post("/api/sessions", json={"title": "Alice"}, headers=ALICE).json()


class TestSessionOwnershipData:
    def test_create_stores_owner(self):
        s = create_session("Chat", owner_id="owner-a")
        assert s["owner_id"] == "owner-a"

    def test_list_filters_by_owner(self):
        mine = create_session("Mine", owner_id="owner-a")
        create_session("Theirs", owner_id="owner-b")
        assert [s["id"] for s in list_sessions(owner_id="owner-a")] == [mine["id"]]

    def test_list_without_owner_returns_all(self):
        create_session("A", owner_id="owner-a")
        create_session("B")
        assert len(list_sessions()) == 2

    def test_init_db_migrates_legacy_sessions_table(self):
        settings.db_path.unlink()
        conn = sqlite3.connect(str(settings.db_path))
        conn.execute(
            "CREATE TABLE sessions (id TEXT PRIMARY KEY, title TEXT NOT NULL, "
            "project_id TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL)"
        )
        conn.close()

        init_db()

        conn = sqlite3.connect(str(settings.db_path))
        columns = {row[1] for row in conn.execute("PRAGMA table_info(sessions)")}
        conn.close()
        assert "owner_id" in columns


class TestSessionIsolationInDemo:
    def test_list_only_shows_own_sessions(self, demo_client, alice_session):
        alice_ids = [s["id"] for s in demo_client.get("/api/sessions", headers=ALICE).json()]
        bob_ids = [s["id"] for s in demo_client.get("/api/sessions", headers=BOB).json()]
        assert alice_ids == [alice_session["id"]]
        assert bob_ids == []

    def test_list_without_client_id_is_empty(self, demo_client, alice_session):
        assert demo_client.get("/api/sessions").json() == []

    def test_create_requires_client_id(self, demo_client):
        assert demo_client.post("/api/sessions", json={"title": "x"}).status_code == 400

    def test_rejects_malformed_client_id(self, demo_client):
        r = demo_client.get("/api/sessions", headers={"X-Client-Id": "bad id!"})
        assert r.status_code == 400

    @pytest.mark.parametrize(
        ("method", "suffix", "body"),
        [
            ("GET", "", None),
            ("GET", "/messages", None),
            ("PATCH", "", {"title": "pwned"}),
            ("DELETE", "", None),
        ],
    )
    def test_other_clients_get_404(self, demo_client, alice_session, method, suffix, body):
        path = f"/api/sessions/{alice_session['id']}{suffix}"
        assert demo_client.request(method, path, json=body, headers=BOB).status_code == 404
        assert demo_client.request("GET", path, headers=ALICE).status_code == 200

    @pytest.mark.parametrize("endpoint", ["/api/ask", "/api/stream"])
    def test_cannot_chat_in_another_clients_session(self, demo_client, alice_session, endpoint):
        body = {**ASK, "session_id": alice_session["id"]}
        assert demo_client.post(endpoint, json=body, headers=BOB).status_code == 404

    def test_owner_can_chat_in_own_session(self, demo_client, alice_session):
        body = {**ASK, "session_id": alice_session["id"]}
        assert demo_client.post("/api/ask", json=body, headers=ALICE).status_code == 200


class TestSessionsOutsideDemo:
    def test_sessions_stay_shared_locally(self, client):
        client.post("/api/sessions", json={"title": "A"}, headers=ALICE)
        assert len(client.get("/api/sessions", headers=BOB).json()) == 1
