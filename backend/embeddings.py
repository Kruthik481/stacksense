"""Text embeddings via fastembed's ONNX build of all-MiniLM-L6-v2 (no torch, small enough for serverless)."""

import logging
from functools import lru_cache

import numpy as np
from config import settings
from fastembed import TextEmbedding

logger = logging.getLogger(__name__)

EMBEDDING_DIM = 384


@lru_cache(maxsize=1)
def get_model() -> TextEmbedding:
    # Loaded on first use so importing the API (and the test suite) never pays for the model.
    logger.info("Loading embedding model %s", settings.embedding_model)
    return TextEmbedding(settings.embedding_model, cache_dir=str(settings.embedding_cache_dir))


def embed(text: str) -> np.ndarray:
    vector = next(iter(get_model().embed([text])))
    return np.asarray(vector, dtype=np.float32)
