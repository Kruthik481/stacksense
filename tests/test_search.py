import numpy as np
from search import BM25, HybridSearcher, reciprocal_rank_fusion


class TestBM25:
    def test_fit_and_score(self):
        docs = [
            "def login(username, password): check credentials",
            "class UserService: get_user delete_user query",
            "def process_data(items): return filtered items",
        ]
        bm25 = BM25().fit(docs)
        scores = bm25.score("login password")
        assert scores[0] > scores[1]
        assert scores[0] > scores[2]

    def test_empty_corpus(self):
        bm25 = BM25().fit([])
        assert bm25.corpus_size == 0

    def test_unknown_terms(self):
        bm25 = BM25().fit(["hello world"])
        scores = bm25.score("xyz abc")
        assert scores[0] == 0.0

    def test_tokenization(self):
        bm25 = BM25()
        tokens = bm25._tokenize("def my_func(arg1, arg2):")
        assert "def" in tokens
        assert "my_func" in tokens
        assert "arg1" in tokens

    def test_idf_weighting(self):
        docs = ["login auth", "login session", "database query"]
        bm25 = BM25().fit(docs)
        scores = bm25.score("auth")
        assert scores[0] > 0
        assert scores[1] == 0
        assert scores[2] == 0

    def test_document_length_normalization(self):
        docs = [
            "login " * 100,
            "login password credentials auth",
        ]
        bm25 = BM25().fit(docs)
        scores = bm25.score("login")
        assert scores[0] > 0
        assert scores[1] > 0
        ratio = scores[0] / scores[1]
        assert ratio < 100


class TestReciprocalRankFusion:
    def test_basic_fusion(self):
        list1 = [0, 1, 2]
        list2 = [2, 0, 1]
        fused = reciprocal_rank_fusion([list1, list2])
        ids = [doc_id for doc_id, _ in fused]
        assert 0 in ids
        assert 2 in ids

    def test_single_list(self):
        fused = reciprocal_rank_fusion([[3, 1, 2]])
        assert fused[0][0] == 3

    def test_disjoint_lists(self):
        fused = reciprocal_rank_fusion([[0, 1], [2, 3]])
        assert len(fused) == 4

    def test_empty_lists(self):
        fused = reciprocal_rank_fusion([[], []])
        assert fused == []


class TestHybridSearcher:
    def test_not_ready_before_build(self):
        searcher = HybridSearcher()
        assert not searcher.is_ready()

    def test_build_and_ready(self, sample_index):
        index, metadata = sample_index
        searcher = HybridSearcher()
        searcher.build(index, metadata)
        assert searcher.is_ready()

    def test_search_returns_results(self, sample_index, fake_embedding):
        index, metadata = sample_index
        searcher = HybridSearcher()
        searcher.build(index, metadata)
        embedding = fake_embedding("login authentication")
        results = searcher.search("login authentication", embedding, k=10, final_k=3)
        assert len(results) > 0
        assert len(results) <= 3

    def test_results_have_scores(self, sample_index, fake_embedding):
        index, metadata = sample_index
        searcher = HybridSearcher()
        searcher.build(index, metadata)
        results = searcher.search("login", fake_embedding("login"), k=10, final_k=3)
        for r in results:
            assert "rrf_score" in r
            assert "bm25_score" in r

    def test_search_not_ready(self):
        searcher = HybridSearcher()
        results = searcher.search("test", np.zeros(384, dtype="float32"), k=5, final_k=3)
        assert results == []

    def test_bm25_boosts_keyword_match(self, sample_index, fake_embedding):
        index, metadata = sample_index
        searcher = HybridSearcher()
        searcher.build(index, metadata)
        results = searcher.search("UserService", fake_embedding("UserService"), k=10, final_k=5)
        names = [r.get("name", "") for r in results]
        assert any("UserService" in n for n in names)
