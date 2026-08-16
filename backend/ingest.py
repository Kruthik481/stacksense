import logging
import os
import pickle
from pathlib import Path

import faiss
import numpy as np
from chunkers import chunk_file
from config import settings
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {
    ".py",
    ".js",
    ".ts",
    ".jsx",
    ".tsx",
    ".java",
    ".cpp",
    ".c",
    ".h",
    ".html",
    ".css",
    ".json",
    ".yaml",
    ".yml",
    ".go",
    ".rs",
    ".rb",
    ".sh",
}

model = SentenceTransformer(settings.embedding_model)


def get_embedding(text: str) -> np.ndarray:
    return np.asarray(model.encode(text))


def load_codebase(path: str) -> list[dict]:
    documents = []
    for root, _, files in os.walk(path):
        for file in files:
            if Path(file).suffix not in SUPPORTED_EXTENSIONS:
                continue
            file_path = os.path.join(root, file)
            try:
                with open(file_path, encoding="utf-8") as f:
                    content = f.read()
                documents.append(
                    {
                        "content": content,
                        "file_name": file,
                        "path": file_path,
                    }
                )
            except (UnicodeDecodeError, PermissionError, OSError) as e:
                logger.warning("Skipping %s: %s", file_path, e)

    logger.info("Loaded %d files from %s", len(documents), path)
    return documents


def create_vector_store(chunks: list[dict]) -> tuple[faiss.Index, list[dict]]:
    dimension = 384
    index = faiss.IndexFlatL2(dimension)
    vectors = []
    metadata = []

    for chunk in chunks:
        emb = get_embedding(chunk["content"])
        vectors.append(emb)
        metadata.append(chunk)

    index.add(np.array(vectors).astype("float32"))
    logger.info("Created FAISS index with %d vectors", len(metadata))
    return index, metadata


def save_index(index: faiss.Index, metadata: list[dict], project_id: str | None = None) -> None:
    if project_id:
        idx_path = settings.project_index_path(project_id)
        meta_path = settings.project_metadata_path(project_id)
    else:
        idx_path = settings.faiss_index_path
        meta_path = settings.metadata_path

    idx_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(idx_path))
    with open(meta_path, "wb") as f:
        pickle.dump(metadata, f)
    logger.info("Saved index to %s", idx_path)


def load_index(project_id: str | None = None) -> tuple[faiss.Index, list[dict]]:
    if project_id:
        idx_path = settings.project_index_path(project_id)
        meta_path = settings.project_metadata_path(project_id)
    else:
        idx_path = settings.faiss_index_path
        meta_path = settings.metadata_path

    index = faiss.read_index(str(idx_path))
    with open(meta_path, "rb") as f:
        metadata = pickle.load(f)
    logger.info("Loaded index with %d vectors from %s", len(metadata), idx_path)
    return index, metadata


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    target = sys.argv[1] if len(sys.argv) > 1 else "sample_repo"
    docs = load_codebase(target)

    all_chunks: list[dict] = []
    for doc in docs:
        all_chunks.extend(chunk_file(doc["content"], doc["file_name"], doc["path"]))

    print(f"Total chunks: {len(all_chunks)}")
    for c in all_chunks:
        label = c.get("name", c["file_name"])
        print(f"  {c['file_name']}:{label} ({c.get('type', '?')}) — {len(c['content'])} chars")

    index, metadata = create_vector_store(all_chunks)
    save_index(index, metadata)
    print(f"Done — {index.ntotal} vectors saved")
