"""Build step for hosted demos: fetch the embedding model and index StackSense's own backend.

Run from anywhere: `python backend/build_demo.py`. Vercel runs it via `[tool.vercel.scripts]`
and the Dockerfile runs it at image build time, so a fresh deployment answers questions
without any manual ingestion.
"""

import logging
import shutil
import tempfile
from pathlib import Path

from chunkers import chunk_file
from config import BASE_DIR, settings
from fastembed import TextEmbedding
from ingest import create_vector_store, load_codebase, save_index

logger = logging.getLogger(__name__)


def export_model(dest: Path) -> None:
    """Download the embedding model and copy it into `dest` as plain files.

    The Hugging Face cache keeps weights in blobs/ and symlinks them from snapshots/.
    Function bundlers follow symlinks, so shipping the cache doubles the model size.
    """
    with tempfile.TemporaryDirectory() as cache:
        TextEmbedding(settings.embedding_model, cache_dir=cache)
        snapshots = sorted(Path(cache).glob("models--*/snapshots/*"))
        if not snapshots:
            raise RuntimeError(f"Model download left no snapshot in {cache}")
        shutil.rmtree(dest, ignore_errors=True)
        shutil.copytree(snapshots[-1], dest, symlinks=False)
    logger.info("Exported %s to %s", settings.embedding_model, dest)


def build_demo_index(source_dir: Path = BASE_DIR) -> int:
    # Store paths relative to the source dir: the build machine and the runtime mount the
    # project at different absolute paths, and the file browser resolves them against BASE_DIR.
    docs = [
        {**doc, "path": Path(doc["path"]).relative_to(source_dir).as_posix()}
        for doc in load_codebase(str(source_dir))
    ]
    if not docs:
        raise RuntimeError(f"No supported files found in {source_dir}")

    chunks = [
        chunk for doc in docs for chunk in chunk_file(doc["content"], doc["file_name"], doc["path"])
    ]
    index, metadata = create_vector_store(chunks)
    save_index(index, metadata)
    logger.info("Demo index built: %d files, %d vectors", len(docs), index.ntotal)
    return int(index.ntotal)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    # Ship the model with the deploy instead of downloading it on every cold start.
    export_model(settings.embedding_bundle_dir)
    build_demo_index()


if __name__ == "__main__":
    main()
