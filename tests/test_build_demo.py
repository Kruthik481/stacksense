import build_demo
from config import settings
from ingest import load_index


def test_indexes_source_with_paths_relative_to_it(tmp_path, monkeypatch, fake_embedding):
    source = tmp_path / "src"
    (source / "pkg").mkdir(parents=True)
    # The chunker drops chunks under 30 chars, so keep these functions realistic in length.
    (source / "auth.py").write_text(
        "def login(user, password):\n    return check(user, password)\n"
    )
    (source / "pkg" / "util.py").write_text(
        "def format_name(first, last):\n    return f'{first} {last}'\n"
    )
    monkeypatch.setattr("ingest.embed", fake_embedding)

    vectors = build_demo.build_demo_index(source)

    index, metadata = load_index()
    assert vectors == index.ntotal == len(metadata) > 0
    paths = {m["path"] for m in metadata}
    assert paths == {"auth.py", "pkg/util.py"}
    # Relative paths resolve against the source dir wherever the build output is deployed.
    assert all((source / p).is_file() for p in paths)
    assert settings.faiss_index_path.exists()
