import logging
import sqlite3
import uuid
from datetime import UTC, datetime

from config import settings

logger = logging.getLogger(__name__)


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(settings.db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db() -> None:
    settings.db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = _get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS projects (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL UNIQUE,
            path TEXT NOT NULL,
            language TEXT,
            files_count INTEGER DEFAULT 0,
            chunks_count INTEGER DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL DEFAULT 'New Chat',
            project_id TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE SET NULL
        );
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
            content TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_messages_session
            ON messages(session_id);
    """)
    conn.close()
    logger.info("Database initialized at %s", settings.db_path)


def create_project(name: str, path: str, language: str = "") -> dict:
    project_id = str(uuid.uuid4())
    now = datetime.now(UTC).isoformat()
    conn = _get_conn()
    conn.execute(
        "INSERT INTO projects (id, name, path, language, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
        (project_id, name, path, language, now, now),
    )
    conn.commit()
    conn.close()
    return {
        "id": project_id,
        "name": name,
        "path": path,
        "language": language,
        "files_count": 0,
        "chunks_count": 0,
        "created_at": now,
        "updated_at": now,
    }


def list_projects() -> list[dict]:
    conn = _get_conn()
    rows = conn.execute("SELECT * FROM projects ORDER BY updated_at DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_project(project_id: str) -> dict | None:
    conn = _get_conn()
    row = conn.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def get_project_by_name(name: str) -> dict | None:
    conn = _get_conn()
    row = conn.execute("SELECT * FROM projects WHERE name = ?", (name,)).fetchone()
    conn.close()
    return dict(row) if row else None


def update_project_stats(project_id: str, files_count: int, chunks_count: int) -> None:
    now = datetime.now(UTC).isoformat()
    conn = _get_conn()
    conn.execute(
        "UPDATE projects SET files_count = ?, chunks_count = ?, updated_at = ? WHERE id = ?",
        (files_count, chunks_count, now, project_id),
    )
    conn.commit()
    conn.close()


def delete_project(project_id: str) -> bool:
    conn = _get_conn()
    cursor = conn.execute("DELETE FROM projects WHERE id = ?", (project_id,))
    conn.commit()
    conn.close()
    return cursor.rowcount > 0


def create_session(title: str = "New Chat", project_id: str | None = None) -> dict:
    session_id = str(uuid.uuid4())
    now = datetime.now(UTC).isoformat()
    conn = _get_conn()
    conn.execute(
        "INSERT INTO sessions (id, title, project_id, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
        (session_id, title, project_id, now, now),
    )
    conn.commit()
    conn.close()
    return {
        "id": session_id,
        "title": title,
        "project_id": project_id,
        "created_at": now,
        "updated_at": now,
    }


def list_sessions() -> list[dict]:
    conn = _get_conn()
    rows = conn.execute("SELECT * FROM sessions ORDER BY updated_at DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_session(session_id: str) -> dict | None:
    conn = _get_conn()
    row = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def delete_session(session_id: str) -> bool:
    conn = _get_conn()
    cursor = conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
    conn.commit()
    conn.close()
    return cursor.rowcount > 0


def update_session_title(session_id: str, title: str) -> None:
    now = datetime.now(UTC).isoformat()
    conn = _get_conn()
    conn.execute(
        "UPDATE sessions SET title = ?, updated_at = ? WHERE id = ?",
        (title, now, session_id),
    )
    conn.commit()
    conn.close()


def add_message(session_id: str, role: str, content: str) -> dict:
    now = datetime.now(UTC).isoformat()
    conn = _get_conn()
    cursor = conn.execute(
        "INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
        (session_id, role, content, now),
    )
    conn.execute("UPDATE sessions SET updated_at = ? WHERE id = ?", (now, session_id))
    conn.commit()
    msg_id = cursor.lastrowid
    conn.close()
    return {
        "id": msg_id,
        "session_id": session_id,
        "role": role,
        "content": content,
        "created_at": now,
    }


def get_messages(session_id: str, limit: int = 20) -> list[dict]:
    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM messages WHERE session_id = ? ORDER BY created_at DESC LIMIT ?",
        (session_id, limit),
    ).fetchall()
    conn.close()
    return [dict(r) for r in reversed(rows)]


def auto_title(session_id: str, first_query: str) -> None:
    title = first_query[:50]
    if len(first_query) > 50:
        title = title.rsplit(" ", 1)[0] + "..."
    update_session_title(session_id, title)
