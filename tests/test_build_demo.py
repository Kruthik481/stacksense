from pathlib import Path

import build_demo
from config import settings
from ingest import load_index


class FakeHubDownload:
    """Mimics the Hugging Face cache: real files in blobs/, symlinks in snapshots/."""

    def __init__(self, model_name: str, cache_dir: str):
        repo = Path(cache_dir) / "models--qdrant--fake-onnx"
        blobs = repo / "blobs"
        snapshot = repo / "snapshots" / "abc123"
        blobs.mkdir(parents=True)
        snapshot.mkdir(parents=True)
        for name, content in {"model.onnx": b"weights", "tokenizer.json": b"{}"}.items():
            (blobs / name).write_bytes(content)
            (snapshot / name).symlink_to(blobs / name)


def test_export_model_writes_real_files_without_cache_duplicates(tmp_path, monkeypatch):
    monkeypatch.setattr(build_demo, "TextEmbedding", FakeHubDownload)
    dest = tmp_path / "bundled"

    build_demo.export_model(dest)

    files = sorted(p.name for p in dest.iterdir())
    assert files == ["model.onnx", "tokenizer.json"]
    # Bundlers follow symlinks, so any left behind would ship the weights twice.
    assert not any(p.is_symlink() for p in dest.iterdir())
    assert (dest / "model.onnx").read_bytes() == b"weights"


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
