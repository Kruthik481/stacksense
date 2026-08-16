from sessions import (
    add_message,
    auto_title,
    create_session,
    delete_session,
    get_messages,
    get_session,
    list_sessions,
    update_session_title,
)


class TestSessionCRUD:
    def test_create_and_get(self):
        s = create_session("My Chat")
        assert s["title"] == "My Chat"
        assert s["id"]

        fetched = get_session(s["id"])
        assert fetched is not None
        assert fetched["title"] == "My Chat"

    def test_list_ordered_by_updated(self):
        s1 = create_session("First")
        create_session("Second")
        update_session_title(s1["id"], "First Updated")

        sessions = list_sessions()
        assert len(sessions) >= 2
        assert sessions[0]["id"] == s1["id"]

    def test_get_nonexistent(self):
        assert get_session("does-not-exist") is None

    def test_delete(self):
        s = create_session("To Delete")
        assert delete_session(s["id"]) is True
        assert get_session(s["id"]) is None

    def test_delete_nonexistent(self):
        assert delete_session("nope") is False

    def test_update_title(self):
        s = create_session("Old Title")
        update_session_title(s["id"], "New Title")
        fetched = get_session(s["id"])
        assert fetched["title"] == "New Title"


class TestMessages:
    def test_add_and_get(self):
        s = create_session()
        add_message(s["id"], "user", "Hello")
        add_message(s["id"], "assistant", "Hi there")

        msgs = get_messages(s["id"])
        assert len(msgs) == 2
        assert msgs[0]["role"] == "user"
        assert msgs[0]["content"] == "Hello"
        assert msgs[1]["role"] == "assistant"

    def test_chronological_order(self):
        s = create_session()
        add_message(s["id"], "user", "first")
        add_message(s["id"], "user", "second")
        add_message(s["id"], "user", "third")

        msgs = get_messages(s["id"])
        assert [m["content"] for m in msgs] == ["first", "second", "third"]

    def test_limit(self):
        s = create_session()
        for i in range(10):
            add_message(s["id"], "user", f"msg {i}")

        msgs = get_messages(s["id"], limit=3)
        assert len(msgs) == 3
        assert msgs[-1]["content"] == "msg 9"

    def test_delete_cascades_messages(self):
        s = create_session()
        add_message(s["id"], "user", "hello")
        delete_session(s["id"])
        msgs = get_messages(s["id"])
        assert len(msgs) == 0

    def test_add_updates_session_timestamp(self):
        s = create_session()
        original = get_session(s["id"])["updated_at"]
        add_message(s["id"], "user", "trigger update")
        updated = get_session(s["id"])["updated_at"]
        assert updated >= original


class TestAutoTitle:
    def test_short_title(self):
        s = create_session()
        auto_title(s["id"], "How does auth work?")
        fetched = get_session(s["id"])
        assert fetched["title"] == "How does auth work?"

    def test_long_title_truncated(self):
        s = create_session()
        long_query = "Explain the entire authentication flow including login logout and session management in detail"
        auto_title(s["id"], long_query)
        fetched = get_session(s["id"])
        assert len(fetched["title"]) <= 53
        assert fetched["title"].endswith("...")
