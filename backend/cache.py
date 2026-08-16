"""LRU + TTL caching for embeddings and query results."""

import hashlib
import logging
import threading
import time
from collections import OrderedDict

import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingCache:
    def __init__(self, maxsize: int = 2048):
        self.maxsize = maxsize
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def _key(self, text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    def get(self, text: str) -> np.ndarray | None:
        key = self._key(text)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self.hits += 1
                return self._cache[key]
            self.misses += 1
            return None

    def put(self, text: str, embedding: np.ndarray) -> None:
        key = self._key(text)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return
            self._cache[key] = embedding
            if len(self._cache) > self.maxsize:
                self._cache.popitem(last=False)

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {
            "size": len(self._cache),
            "maxsize": self.maxsize,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 4) if total > 0 else 0,
        }

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            self.hits = 0
            self.misses = 0


class QueryCache:
    def __init__(self, maxsize: int = 256, ttl_seconds: int = 300):
        self.maxsize = maxsize
        self.ttl = ttl_seconds
        self._cache: OrderedDict[str, tuple[float, list[dict]]] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def _key(self, query: str, project_id: str | None) -> str:
        raw = f"{project_id or 'default'}:{query}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def get(self, query: str, project_id: str | None = None) -> list[dict] | None:
        key = self._key(query, project_id)
        with self._lock:
            if key in self._cache:
                ts, results = self._cache[key]
                if time.time() - ts < self.ttl:
                    self._cache.move_to_end(key)
                    self.hits += 1
                    return results
                del self._cache[key]
            self.misses += 1
            return None

    def put(self, query: str, results: list[dict], project_id: str | None = None) -> None:
        key = self._key(query, project_id)
        with self._lock:
            self._cache[key] = (time.time(), results)
            self._cache.move_to_end(key)
            if len(self._cache) > self.maxsize:
                self._cache.popitem(last=False)

    def invalidate(self, project_id: str | None = None) -> None:
        with self._lock:
            [k for k, (_, _) in self._cache.items()] if project_id is None else []
            if project_id is None:
                self._cache.clear()
            else:
                for key in list(self._cache.keys()):
                    self._cache.pop(key, None)

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {
            "size": len(self._cache),
            "maxsize": self.maxsize,
            "ttl_seconds": self.ttl,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 4) if total > 0 else 0,
        }

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            self.hits = 0
            self.misses = 0


embedding_cache = EmbeddingCache()
query_cache = QueryCache()
