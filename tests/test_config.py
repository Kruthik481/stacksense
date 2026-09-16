from pathlib import Path

import pytest
from config import BASE_DIR, Settings


@pytest.fixture
def clean_env(monkeypatch):
    for var in ("VERCEL", "DB_PATH", "PUBLIC_DEMO"):
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


def make_settings() -> Settings:
    return Settings(_env_file=None)


class TestLocalDefaults:
    def test_db_lives_in_backend_data(self, clean_env):
        assert make_settings().db_path == BASE_DIR / "data" / "sessions.db"

    def test_public_demo_off(self, clean_env):
        assert make_settings().public_demo is False


class TestVercelDefaults:
    def test_db_moves_to_writable_tmp(self, clean_env):
        clean_env.setenv("VERCEL", "1")
        assert make_settings().db_path == Path("/tmp/stacksense/sessions.db")

    def test_public_demo_on_so_filesystem_endpoints_are_locked(self, clean_env):
        clean_env.setenv("VERCEL", "1")
        assert make_settings().public_demo is True

    def test_explicit_env_vars_still_win(self, clean_env):
        clean_env.setenv("VERCEL", "1")
        clean_env.setenv("PUBLIC_DEMO", "false")
        clean_env.setenv("DB_PATH", "/tmp/custom.db")
        settings = make_settings()
        assert settings.public_demo is False
        assert settings.db_path == Path("/tmp/custom.db")
