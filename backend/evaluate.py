"""Retrieval evaluation harness — measures search quality with precision@k and MRR."""

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import faiss
import numpy as np
from search import HybridSearcher

logger = logging.getLogger(__name__)

EVAL_SET = [
    {
        "query": "How does the login function work?",
        "expected_files": ["auth.py"],
        "expected_names": ["login"],
    },
    {
        "query": "Show me the UserService class",
        "expected_files": ["users.py"],
        "expected_names": ["UserService"],
    },
    {
        "query": "How is data processed?",
        "expected_files": ["utils.py"],
        "expected_names": ["process_data"],
    },
    {
        "query": "What are the API endpoints?",
        "expected_files": ["main.py", "routes.py"],
        "expected_names": [],
    },
    {
        "query": "How does authentication work?",
        "expected_files": ["auth.py"],
        "expected_names": ["login"],
    },
    {
        "query": "database query methods",
        "expected_files": ["users.py"],
        "expected_names": ["UserService", "get_user"],
    },
    {
        "query": "delete user",
        "expected_files": ["users.py"],
        "expected_names": ["delete_user"],
    },
    {
        "query": "error handling patterns",
        "expected_files": [],
        "expected_names": [],
    },
]


@dataclass
class QueryMetrics:
    query: str
    precision_at_1: float = 0.0
    precision_at_3: float = 0.0
    precision_at_5: float = 0.0
    mrr: float = 0.0
    num_results: int = 0
    latency_ms: float = 0.0
    retrieved_files: list[str] = field(default_factory=list)
    retrieved_names: list[str] = field(default_factory=list)


@dataclass
class EvalReport:
    timestamp: str = ""
    num_queries: int = 0
    mean_precision_at_1: float = 0.0
    mean_precision_at_3: float = 0.0
    mean_precision_at_5: float = 0.0
    mean_mrr: float = 0.0
    mean_latency_ms: float = 0.0
    query_details: list[QueryMetrics] = field(default_factory=list)


def _is_relevant(result: dict, expected_files: list[str], expected_names: list[str]) -> bool:
    file_match = (
        any(ef in result.get("file_name", "") for ef in expected_files) if expected_files else False
    )

    name_match = (
        any(
            en.lower() in result.get("name", "").lower()
            or en.lower() in result.get("content", "").lower()
            for en in expected_names
        )
        if expected_names
        else False
    )

    return file_match or name_match


def precision_at_k(
    results: list[dict], expected_files: list[str], expected_names: list[str], k: int
) -> float:
    if not expected_files and not expected_names:
        return 0.0
    top_k = results[:k]
    if not top_k:
        return 0.0
    relevant = sum(1 for r in top_k if _is_relevant(r, expected_files, expected_names))
    return relevant / k


def mean_reciprocal_rank(
    results: list[dict], expected_files: list[str], expected_names: list[str]
) -> float:
    if not expected_files and not expected_names:
        return 0.0
    for i, r in enumerate(results):
        if _is_relevant(r, expected_files, expected_names):
            return 1.0 / (i + 1)
    return 0.0


def evaluate_retrieval(
    index: faiss.Index,
    metadata: list[dict],
    get_embedding_fn,
    searcher: HybridSearcher | None = None,
    eval_set: list[dict] | None = None,
) -> EvalReport:
    if eval_set is None:
        eval_set = EVAL_SET

    report = EvalReport(
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        num_queries=len(eval_set),
    )

    for item in eval_set:
        query = str(item["query"])
        expected_files = list(item.get("expected_files", []))
        expected_names = list(item.get("expected_names", []))

        start = time.perf_counter()

        if searcher and searcher.is_ready():
            embedding = get_embedding_fn(query)
            results = searcher.search(query, embedding, k=20, final_k=10)
        else:
            embedding = get_embedding_fn(query)
            vec = np.array([embedding]).astype("float32")
            distances, indices = index.search(vec, 10)
            results = [
                metadata[idx]
                for dist, idx in zip(distances[0], indices[0], strict=False)
                if idx != -1
            ]

        latency = (time.perf_counter() - start) * 1000

        qm = QueryMetrics(
            query=query,
            precision_at_1=precision_at_k(results, expected_files, expected_names, 1),
            precision_at_3=precision_at_k(results, expected_files, expected_names, 3),
            precision_at_5=precision_at_k(results, expected_files, expected_names, 5),
            mrr=mean_reciprocal_rank(results, expected_files, expected_names),
            num_results=len(results),
            latency_ms=round(latency, 2),
            retrieved_files=[r.get("file_name", "") for r in results[:5]],
            retrieved_names=[r.get("name", "") for r in results[:5]],
        )
        report.query_details.append(qm)

    [
        q
        for q in report.query_details
        if q.mrr > 0 or q.precision_at_1 > 0 or any(report.query_details)
    ]

    n = len(report.query_details)
    if n > 0:
        report.mean_precision_at_1 = round(
            sum(q.precision_at_1 for q in report.query_details) / n, 4
        )
        report.mean_precision_at_3 = round(
            sum(q.precision_at_3 for q in report.query_details) / n, 4
        )
        report.mean_precision_at_5 = round(
            sum(q.precision_at_5 for q in report.query_details) / n, 4
        )
        report.mean_mrr = round(sum(q.mrr for q in report.query_details) / n, 4)
        report.mean_latency_ms = round(sum(q.latency_ms for q in report.query_details) / n, 2)

    return report


def report_to_dict(report: EvalReport) -> dict:
    d = asdict(report)
    return d


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from ingest import load_index
    from query import get_embedding
    from search import HybridSearcher

    try:
        index, metadata = load_index()
    except (FileNotFoundError, RuntimeError):
        print("No index found. Run ingest first.")
        raise SystemExit(1) from None

    searcher = HybridSearcher()
    searcher.build(index, metadata)

    print("\n=== Vector-only search ===")
    vector_report = evaluate_retrieval(index, metadata, get_embedding)
    print(f"  P@1: {vector_report.mean_precision_at_1}")
    print(f"  P@3: {vector_report.mean_precision_at_3}")
    print(f"  MRR: {vector_report.mean_mrr}")
    print(f"  Latency: {vector_report.mean_latency_ms}ms")

    print("\n=== Hybrid search (BM25 + FAISS + RRF) ===")
    hybrid_report = evaluate_retrieval(index, metadata, get_embedding, searcher)
    print(f"  P@1: {hybrid_report.mean_precision_at_1}")
    print(f"  P@3: {hybrid_report.mean_precision_at_3}")
    print(f"  MRR: {hybrid_report.mean_mrr}")
    print(f"  Latency: {hybrid_report.mean_latency_ms}ms")

    improvement = hybrid_report.mean_mrr - vector_report.mean_mrr
    print(f"\n  MRR improvement: {improvement:+.4f}")

    out = Path("data/eval_report.json")
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as f:
        json.dump(
            {"vector_only": report_to_dict(vector_report), "hybrid": report_to_dict(hybrid_report)},
            f,
            indent=2,
        )
    print(f"\nFull report saved to {out}")
