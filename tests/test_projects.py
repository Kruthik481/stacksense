from sessions import (
    create_project,
    create_session,
    delete_project,
    get_project,
    get_project_by_name,
    get_session,
    list_projects,
    update_project_stats,
)


class TestProjectCRUD:
    def test_create_and_get(self):
        p = create_project("my-app", "/tmp/my-app", "python")
        assert p["name"] == "my-app"
        assert p["path"] == "/tmp/my-app"
        assert p["language"] == "python"
        assert p["id"]

        fetched = get_project(p["id"])
        assert fetched is not None
        assert fetched["name"] == "my-app"

    def test_get_by_name(self):
        create_project("web-ui", "/tmp/web-ui")
        p = get_project_by_name("web-ui")
        assert p is not None
        assert p["name"] == "web-ui"

    def test_get_nonexistent(self):
        assert get_project("does-not-exist") is None
        assert get_project_by_name("nope") is None

    def test_list_projects(self):
        create_project("proj-a", "/tmp/a")
        create_project("proj-b", "/tmp/b")
        projects = list_projects()
        names = {p["name"] for p in projects}
        assert "proj-a" in names
        assert "proj-b" in names

    def test_delete_project(self):
        p = create_project("to-delete", "/tmp/del")
        assert delete_project(p["id"])
        assert get_project(p["id"]) is None

    def test_delete_nonexistent(self):
        assert not delete_project("nope")

    def test_update_stats(self):
        p = create_project("stats-test", "/tmp/stats")
        update_project_stats(p["id"], 10, 42)
        fetched = get_project(p["id"])
        assert fetched["files_count"] == 10
        assert fetched["chunks_count"] == 42


class TestSessionWithProject:
    def test_create_session_with_project(self):
        p = create_project("session-proj", "/tmp/sp")
        s = create_session("Chat", project_id=p["id"])
        assert s["project_id"] == p["id"]
        fetched = get_session(s["id"])
        assert fetched["project_id"] == p["id"]

    def test_create_session_without_project(self):
        s = create_session("Plain Chat")
        assert s["project_id"] is None

    def test_delete_project_nulls_session(self):
        p = create_project("del-proj", "/tmp/dp")
        s = create_session("Chat", project_id=p["id"])
        delete_project(p["id"])
        fetched = get_session(s["id"])
        assert fetched is not None
        assert fetched["project_id"] is None
