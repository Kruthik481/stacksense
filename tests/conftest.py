import os
import tempfile

_test_dir = tempfile.mkdtemp(prefix="stacksense_test_")
os.environ["DB_PATH"] = os.path.join(_test_dir, "test.db")
os.environ["FAISS_INDEX_PATH"] = os.path.join(_test_dir, "test.faiss")
os.environ["METADATA_PATH"] = os.path.join(_test_dir, "test.pkl")

import hashlib  # noqa: E402

import faiss  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from config import settings  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sessions import init_db  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_db():
    if settings.db_path.exists():
        settings.db_path.unlink()
    init_db()
    yield
    if settings.db_path.exists():
        settings.db_path.unlink()


@pytest.fixture
def fake_embedding():
    def _embed(text: str) -> np.ndarray:
        h = hashlib.md5(text.encode()).digest()
        rng = np.random.RandomState(int.from_bytes(h[:4], "big"))
        return rng.randn(384).astype("float32")

    return _embed


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
    {
        "content": "def process_data(items):\n    return [x * 2 for x in items if x > 0]",
        "file_name": "utils.py",
        "path": "/app/utils.py",
        "type": "FunctionDef",
        "name": "process_data",
        "start_line": 1,
        "end_line": 2,
    },
]


@pytest.fixture
def sample_index(fake_embedding):
    dim = 384
    index = faiss.IndexFlatL2(dim)
    vectors = np.array([fake_embedding(c["content"]) for c in SAMPLE_CHUNKS]).astype("float32")
    index.add(vectors)
    return index, list(SAMPLE_CHUNKS)


@pytest.fixture
def mock_ollama(monkeypatch):
    def fake_chat(**kwargs):
        if kwargs.get("stream"):

            def gen():
                for word in ["The ", "login ", "function ", "checks ", "credentials."]:
                    yield {"message": {"content": word}}

            return gen()
        return {"message": {"content": "The login function checks credentials."}}

    monkeypatch.setattr("ollama.chat", fake_chat)


@pytest.fixture
def client(sample_index, mock_ollama):
    from main import app, store

    store.index, store.metadata = sample_index
    return TestClient(app)
