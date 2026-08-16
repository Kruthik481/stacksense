import logging
import math
import re
from collections import Counter

import faiss
import numpy as np

logger = logging.getLogger(__name__)


class BM25:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_freqs: dict[str, int] = {}
        self.doc_lens: list[int] = []
        self.avg_dl: float = 0
        self.corpus_size: int = 0
        self.doc_term_freqs: list[dict[str, int]] = []

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"[a-zA-Z_]\w*", text.lower())

    def fit(self, documents: list[str]) -> "BM25":
        self.corpus_size = len(documents)
        self.doc_term_freqs = []
        self.doc_lens = []
        self.doc_freqs = {}

        for doc in documents:
            tokens = self._tokenize(doc)
            self.doc_lens.append(len(tokens))
            tf = Counter(tokens)
            self.doc_term_freqs.append(dict(tf))
            for term in tf:
                self.doc_freqs[term] = self.doc_freqs.get(term, 0) + 1

        self.avg_dl = sum(self.doc_lens) / self.corpus_size if self.corpus_size else 1
        return self

    def score(self, query: str) -> list[float]:
        tokens = self._tokenize(query)
        scores = [0.0] * self.corpus_size

        for term in tokens:
            if term not in self.doc_freqs:
                continue
            df = self.doc_freqs[term]
            idf = math.log((self.corpus_size - df + 0.5) / (df + 0.5) + 1)

            for i in range(self.corpus_size):
                tf = self.doc_term_freqs[i].get(term, 0)
                if tf == 0:
                    continue
                dl = self.doc_lens[i]
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * dl / self.avg_dl)
                scores[i] += idf * numerator / denominator

        return scores


def reciprocal_rank_fusion(ranked_lists: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    scores: dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked):
            scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


class HybridSearcher:
    def __init__(self):
        self.bm25: BM25 | None = None
        self.index: faiss.Index | None = None
        self.metadata: list[dict] | None = None

    def build(self, index: faiss.Index, metadata: list[dict]) -> None:
        self.index = index
        self.metadata = metadata
        documents = [m["content"] for m in metadata]
        self.bm25 = BM25().fit(documents)
        logger.info("HybridSearcher built with %d documents", len(documents))

    def is_ready(self) -> bool:
        return all([self.bm25, self.index, self.metadata])

    def search(
        self,
        query_text: str,
        query_embedding: np.ndarray,
        k: int = 20,
        final_k: int = 5,
    ) -> list[dict]:
        if not self.is_ready():
            return []

        vec = np.array([query_embedding]).astype("float32")
        distances, indices = self.index.search(vec, k)
        vector_ranked = [
            int(idx)
            for idx, dist in zip(indices[0], distances[0], strict=False)
            if idx != -1 and dist < 2.0
        ]

        bm25_scores = self.bm25.score(query_text)
        bm25_ranked = sorted(
            range(len(bm25_scores)),
            key=lambda i: bm25_scores[i],
            reverse=True,
        )[:k]

        fused = reciprocal_rank_fusion([vector_ranked, bm25_ranked])

        results = []
        for doc_id, rrf_score in fused[:final_k]:
            if doc_id < len(self.metadata):
                entry = dict(self.metadata[doc_id])
                entry["rrf_score"] = round(rrf_score, 4)
                entry["bm25_score"] = round(bm25_scores[doc_id], 4)
                results.append(entry)

        return results
