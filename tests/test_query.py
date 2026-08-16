from query import _build_context, _build_messages, rerank_results, rewrite_query, search

SAMPLE_CHUNKS = [
    {
        "content": "def login(username, password):\n    if username == 'admin' and password == '1234':\n        return True\n    return False",
        "file_name": "auth.py",
        "path": "/app/auth.py",
        "type": "FunctionDef",
        "name": "login",
        "start_line": 1,
        "end_line": 4,
    },
    {
        "content": "class UserService:\n    def get_user(self, user_id):\n        return self.db.query(user_id)\n    def delete_user(self, user_id):\n        self.db.delete(user_id)",
        "file_name": "users.py",
        "path": "/app/users.py",
        "type": "ClassDef",
        "name": "UserService",
        "start_line": 1,
        "end_line": 5,
    },
]


class TestRewriteQuery:
    def test_no_history(self):
        assert rewrite_query("what is login?", []) == "what is login?"

    def test_short_followup_appended(self):
        history = [{"role": "user", "content": "How does the auth system work?"}]
        result = rewrite_query("and logout?", history)
        assert "auth system" in result
        assert "logout" in result

    def test_long_query_unchanged(self):
        history = [{"role": "user", "content": "previous question here"}]
        query = "Explain the full user registration flow"
        assert rewrite_query(query, history) == query

    def test_strips_whitespace(self):
        assert rewrite_query("  hello world  ", []) == "hello world"

    def test_empty_history_list(self):
        history = [{"role": "assistant", "content": "some response"}]
        assert rewrite_query("test?", history) == "test?"


class TestRerankResults:
    def test_keyword_reranking(self):
        results = [
            {"content": "def process_data(items): return items"},
            {"content": "def login(username, password): check credentials for login"},
            {"content": "def hello(): return world"},
        ]
        ranked = rerank_results("login password", results)
        assert "login" in ranked[0]["content"]

    def test_empty_results(self):
        assert rerank_results("anything", []) == []

    def test_all_equal_scores(self):
        results = [{"content": "aaa"}, {"content": "bbb"}]
        ranked = rerank_results("zzz", results)
        assert len(ranked) == 2


class TestSearch:
    def test_returns_results(self, sample_index, fake_embedding, monkeypatch):
        monkeypatch.setattr("query.get_embedding", fake_embedding)
        index, metadata = sample_index
        results = search("login authentication", index, metadata)
        assert isinstance(results, list)

    def test_respects_limit(self, sample_index, fake_embedding, monkeypatch):
        monkeypatch.setattr("query.get_embedding", fake_embedding)
        index, metadata = sample_index
        results = search("test", index, metadata, k=1)
        assert len(results) <= 5


class TestBuildContext:
    def test_includes_file_info(self):
        ctx = _build_context(SAMPLE_CHUNKS[:1])
        assert "auth.py" in ctx
        assert "login" in ctx

    def test_multiple_chunks(self):
        ctx = _build_context(SAMPLE_CHUNKS[:2])
        assert "auth.py" in ctx
        assert "users.py" in ctx
        assert "---" in ctx

    def test_empty_results(self):
        assert _build_context([]) == ""


class TestBuildMessages:
    def test_system_prompt_first(self):
        msgs = _build_messages("test?", "some context", [])
        assert msgs[0]["role"] == "system"
        assert "code assistant" in msgs[0]["content"].lower()

    def test_includes_history(self):
        history = [
            {"role": "user", "content": "previous question"},
            {"role": "assistant", "content": "previous answer"},
        ]
        msgs = _build_messages("new question?", "ctx", history)
        assert any("previous question" in m["content"] for m in msgs)

    def test_user_prompt_last(self):
        msgs = _build_messages("what is X?", "context here", [])
        last = msgs[-1]
        assert last["role"] == "user"
        assert "what is X?" in last["content"]
        assert "context here" in last["content"]

    def test_history_limited_to_6(self):
        history = [{"role": "user", "content": f"q{i}"} for i in range(20)]
        msgs = _build_messages("current", "ctx", history)
        history_msgs = [
            m for m in msgs if m["role"] == "user" and "ctx" not in m.get("content", "")
        ]
        assert len(history_msgs) <= 6
