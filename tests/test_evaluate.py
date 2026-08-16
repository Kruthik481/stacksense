import pytest
from evaluate import (
    _is_relevant,
    evaluate_retrieval,
    mean_reciprocal_rank,
    precision_at_k,
    report_to_dict,
)

SAMPLE_RESULTS = [
    {"file_name": "auth.py", "name": "login", "content": "def login(username, password): ..."},
    {"file_name": "users.py", "name": "UserService", "content": "class UserService: ..."},
    {"file_name": "utils.py", "name": "process_data", "content": "def process_data(items): ..."},
]


class TestIsRelevant:
    def test_file_match(self):
        assert _is_relevant(SAMPLE_RESULTS[0], ["auth.py"], [])

    def test_name_match(self):
        assert _is_relevant(SAMPLE_RESULTS[0], [], ["login"])

    def test_no_match(self):
        assert not _is_relevant(SAMPLE_RESULTS[0], ["other.py"], ["other_func"])

    def test_empty_expected(self):
        assert not _is_relevant(SAMPLE_RESULTS[0], [], [])


class TestPrecisionAtK:
    def test_perfect_at_1(self):
        assert precision_at_k(SAMPLE_RESULTS, ["auth.py"], ["login"], 1) == 1.0

    def test_partial_at_3(self):
        p = precision_at_k(SAMPLE_RESULTS, ["auth.py"], ["login"], 3)
        assert 0 < p <= 1.0

    def test_no_relevant(self):
        assert precision_at_k(SAMPLE_RESULTS, ["missing.py"], [], 3) == 0.0

    def test_empty_expected(self):
        assert precision_at_k(SAMPLE_RESULTS, [], [], 3) == 0.0

    def test_empty_results(self):
        assert precision_at_k([], ["auth.py"], ["login"], 3) == 0.0


class TestMRR:
    def test_first_relevant(self):
        assert mean_reciprocal_rank(SAMPLE_RESULTS, ["auth.py"], []) == 1.0

    def test_second_relevant(self):
        assert mean_reciprocal_rank(SAMPLE_RESULTS, ["users.py"], []) == 0.5

    def test_no_relevant(self):
        assert mean_reciprocal_rank(SAMPLE_RESULTS, ["missing.py"], []) == 0.0

    def test_name_match_mrr(self):
        mrr = mean_reciprocal_rank(SAMPLE_RESULTS, [], ["process_data"])
        assert mrr == pytest.approx(1 / 3)


class TestEvaluateRetrieval:
    def test_runs_with_sample_index(self, sample_index, fake_embedding):
        index, metadata = sample_index
        report = evaluate_retrieval(index, metadata, fake_embedding)
        assert report.num_queries > 0
        assert 0 <= report.mean_mrr <= 1
        assert 0 <= report.mean_precision_at_1 <= 1
        assert len(report.query_details) == report.num_queries

    def test_custom_eval_set(self, sample_index, fake_embedding):
        index, metadata = sample_index
        eval_set = [
            {"query": "login", "expected_files": ["auth.py"], "expected_names": ["login"]},
        ]
        report = evaluate_retrieval(index, metadata, fake_embedding, eval_set=eval_set)
        assert report.num_queries == 1

    def test_report_to_dict(self, sample_index, fake_embedding):
        index, metadata = sample_index
        report = evaluate_retrieval(index, metadata, fake_embedding)
        d = report_to_dict(report)
        assert isinstance(d, dict)
        assert "mean_mrr" in d
        assert "query_details" in d
        assert isinstance(d["query_details"], list)

    def test_latency_recorded(self, sample_index, fake_embedding):
        index, metadata = sample_index
        report = evaluate_retrieval(index, metadata, fake_embedding)
        for qm in report.query_details:
            assert qm.latency_ms >= 0

    def test_with_hybrid_searcher(self, sample_index, fake_embedding):
        from search import HybridSearcher

        index, metadata = sample_index
        searcher = HybridSearcher()
        searcher.build(index, metadata)
        report = evaluate_retrieval(index, metadata, fake_embedding, searcher=searcher)
        assert report.num_queries > 0
        assert report.mean_latency_ms >= 0
