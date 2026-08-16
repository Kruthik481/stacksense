import os
import tempfile
import time

import numpy as np
import pytest
from cache import EmbeddingCache, QueryCache
from depgraph import (
    DependencyGraph,
    parse_go_imports,
    parse_js_imports,
    parse_python_imports,
)

# =================== Embedding Cache ===================


class TestEmbeddingCache:
    def test_put_and_get(self):
        cache = EmbeddingCache(maxsize=16)
        vec = np.array([1.0, 2.0, 3.0], dtype="float32")
        cache.put("hello", vec)
        result = cache.get("hello")
        assert result is not None
        np.testing.assert_array_equal(result, vec)

    def test_miss_returns_none(self):
        cache = EmbeddingCache(maxsize=16)
        assert cache.get("missing") is None

    def test_lru_eviction(self):
        cache = EmbeddingCache(maxsize=3)
        for i in range(4):
            cache.put(f"item{i}", np.array([float(i)]))
        assert cache.get("item0") is None
        assert cache.get("item3") is not None

    def test_hit_miss_stats(self):
        cache = EmbeddingCache(maxsize=16)
        cache.put("a", np.array([1.0]))
        cache.get("a")
        cache.get("b")
        stats = cache.stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5
        assert stats["size"] == 1

    def test_clear_resets(self):
        cache = EmbeddingCache(maxsize=16)
        cache.put("x", np.array([1.0]))
        cache.get("x")
        cache.clear()
        assert cache.stats()["size"] == 0
        assert cache.stats()["hits"] == 0
        assert cache.get("x") is None

    def test_duplicate_put_no_growth(self):
        cache = EmbeddingCache(maxsize=16)
        cache.put("same", np.array([1.0]))
        cache.put("same", np.array([2.0]))
        assert cache.stats()["size"] == 1


# =================== Query Cache ===================


class TestQueryCache:
    def test_put_and_get(self):
        cache = QueryCache(maxsize=16, ttl_seconds=60)
        results = [{"file": "a.py", "content": "hello"}]
        cache.put("search query", results, "proj1")
        got = cache.get("search query", "proj1")
        assert got == results

    def test_miss_returns_none(self):
        cache = QueryCache(maxsize=16, ttl_seconds=60)
        assert cache.get("nothing") is None

    def test_ttl_expiry(self):
        cache = QueryCache(maxsize=16, ttl_seconds=0)
        cache.put("q", [{"file": "b.py"}])
        time.sleep(0.05)
        assert cache.get("q") is None

    def test_project_scoping(self):
        cache = QueryCache(maxsize=16, ttl_seconds=60)
        cache.put("q", [{"a": 1}], "proj_a")
        cache.put("q", [{"b": 2}], "proj_b")
        assert cache.get("q", "proj_a") == [{"a": 1}]
        assert cache.get("q", "proj_b") == [{"b": 2}]

    def test_lru_eviction(self):
        cache = QueryCache(maxsize=2, ttl_seconds=60)
        cache.put("q1", [{"f": 1}])
        cache.put("q2", [{"f": 2}])
        cache.put("q3", [{"f": 3}])
        assert cache.get("q1") is None
        assert cache.get("q3") is not None

    def test_clear_resets(self):
        cache = QueryCache(maxsize=16, ttl_seconds=60)
        cache.put("q", [{"f": 1}])
        cache.get("q")
        cache.clear()
        assert cache.stats()["size"] == 0
        assert cache.stats()["hits"] == 0

    def test_stats(self):
        cache = QueryCache(maxsize=16, ttl_seconds=300)
        cache.put("q", [])
        cache.get("q")
        cache.get("miss")
        stats = cache.stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["ttl_seconds"] == 300


# =================== Python Imports ===================


class TestParsePythonImports:
    def test_import_statement(self):
        code = "import os\nimport sys"
        result = parse_python_imports(code)
        assert "os" in result
        assert "sys" in result

    def test_from_import(self):
        code = "from pathlib import Path\nfrom os.path import join"
        result = parse_python_imports(code)
        assert "pathlib" in result
        assert "os.path" in result

    def test_syntax_error_fallback(self):
        code = "import os\nthis is {{ invalid python"
        result = parse_python_imports(code)
        assert "os" in result

    def test_empty(self):
        assert parse_python_imports("x = 1") == []


# =================== JS Imports ===================


class TestParseJSImports:
    def test_es6_import(self):
        code = "import React from 'react';\nimport { useState } from 'react';"
        result = parse_js_imports(code)
        assert "react" in result

    def test_require(self):
        code = "const fs = require('fs');"
        result = parse_js_imports(code)
        assert "fs" in result

    def test_export_from(self):
        code = "export { default } from './utils';"
        result = parse_js_imports(code)
        assert "./utils" in result

    def test_dynamic_import(self):
        code = "const mod = import('./lazy');"
        result = parse_js_imports(code)
        assert "./lazy" in result


# =================== Go Imports ===================


class TestParseGoImports:
    def test_standard_imports(self):
        code = 'import (\n\t"fmt"\n\t"net/http"\n)'
        result = parse_go_imports(code)
        assert "fmt" in result
        assert "net/http" in result


# =================== Dependency Graph ===================


class TestDependencyGraph:
    @pytest.fixture
    def sample_docs(self):
        return [
            {
                "path": "/app/main.py",
                "file_name": "main.py",
                "content": "from auth import login\nfrom utils import process_data",
            },
            {
                "path": "/app/auth.py",
                "file_name": "auth.py",
                "content": "def login():\n    pass",
            },
            {
                "path": "/app/utils.py",
                "file_name": "utils.py",
                "content": "def process_data():\n    pass",
            },
        ]

    def test_build(self, sample_docs):
        g = DependencyGraph()
        g.build(sample_docs)
        assert len(g.files) == 3

    def test_dependencies(self, sample_docs):
        g = DependencyGraph()
        g.build(sample_docs)
        deps = g.get_dependencies("/app/main.py")
        assert len(deps) == 2

    def test_dependents(self, sample_docs):
        g = DependencyGraph()
        g.build(sample_docs)
        dependents = g.get_dependents("/app/auth.py")
        assert "/app/main.py" in dependents

    def test_related_files(self, sample_docs):
        g = DependencyGraph()
        g.build(sample_docs)
        related = g.get_related_files("/app/auth.py", depth=1)
        assert "/app/main.py" in related

    def test_to_dict(self, sample_docs):
        g = DependencyGraph()
        g.build(sample_docs)
        d = g.to_dict()
        assert "nodes" in d
        assert "total_files" in d
        assert "total_edges" in d
        assert d["total_files"] == 3
        assert d["total_edges"] >= 2

    def test_empty_graph(self):
        g = DependencyGraph()
        d = g.to_dict()
        assert d["total_files"] == 0
        assert d["nodes"] == []

    def test_no_self_edges(self, sample_docs):
        g = DependencyGraph()
        g.build(sample_docs)
        for f in g.files:
            assert f not in g.edges.get(f, set())


# =================== API: Cache Stats ===================


class TestCacheStatsEndpoint:
    def test_returns_stats(self, client):
        r = client.get("/api/cache/stats")
        assert r.status_code == 200
        data = r.json()
        assert "embedding_cache" in data
        assert "query_cache" in data
        assert "size" in data["embedding_cache"]
        assert "hit_rate" in data["embedding_cache"]
        assert "ttl_seconds" in data["query_cache"]


# =================== API: Files ===================


class TestFilesEndpoint:
    def test_list_files(self, client):
        r = client.get("/api/files")
        assert r.status_code == 200
        files = r.json()
        assert isinstance(files, list)
        assert len(files) == 3
        paths = [f["path"] for f in files]
        assert "/app/auth.py" in paths

    def test_file_has_chunks(self, client):
        r = client.get("/api/files")
        files = r.json()
        for f in files:
            assert "chunks" in f
            assert f["chunks"] >= 1

    def test_no_index_returns_503(self, client):
        from main import store

        orig = store.index, store.metadata
        store.index, store.metadata = None, None
        r = client.get("/api/files")
        assert r.status_code == 503
        store.index, store.metadata = orig


# =================== API: File Source ===================


class TestFileSourceEndpoint:
    def test_existing_file_returns_source(self, client):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("print('hello')\n")
            path = os.path.realpath(f.name)

        from main import store

        store.metadata.append({"path": path, "file_name": os.path.basename(path)})
        try:
            r = client.get(f"/api/files/{path}/source")
            assert r.status_code == 200
            data = r.json()
            assert "content" in data
            assert "print('hello')" in data["content"]
            assert data["lines"] >= 1
        finally:
            store.metadata.pop()
            os.unlink(path)

    def test_non_indexed_file_returns_403(self, client):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("secret")
            path = f.name
        try:
            r = client.get(f"/api/files/{path}/source")
            assert r.status_code == 403
        finally:
            os.unlink(path)

    def test_missing_file_returns_403(self, client):
        r = client.get("/api/files/%2Fno%2Fsuch%2Ffile.py/source")
        assert r.status_code == 403


# =================== API: Dependency Graph ===================


class TestGraphEndpoint:
    def test_returns_graph(self, client):
        r = client.get("/api/graph")
        assert r.status_code == 200
        data = r.json()
        assert "nodes" in data
        assert "total_files" in data
        assert "total_edges" in data


# =================== API: Background Ingestion ===================


class TestBackgroundIngestion:
    def test_start_returns_job_id(self, client, tmp_path):
        py_file = tmp_path / "example.py"
        py_file.write_text("x = 1\n")
        r = client.post("/api/ingest/start", json={"path": str(tmp_path)})
        assert r.status_code == 200
        data = r.json()
        assert "job_id" in data
        assert data["status"] == "started"

    def test_progress_stream(self, client, tmp_path):
        py_file = tmp_path / "example.py"
        py_file.write_text("x = 1\n")
        start = client.post("/api/ingest/start", json={"path": str(tmp_path)}).json()
        job_id = start["job_id"]

        time.sleep(2)

        r = client.get(f"/api/ingest/{job_id}/progress")
        assert r.status_code == 200
        assert "text/event-stream" in r.headers["content-type"]
        body = r.text
        assert "loading" in body or "done" in body or "result" in body

    def test_bad_path_returns_400(self, client):
        r = client.post("/api/ingest/start", json={"path": "/no/such/path"})
        assert r.status_code == 400

    def test_nonexistent_job_returns_404(self, client):
        r = client.get("/api/ingest/nope/progress")
        assert r.status_code == 404


# =================== Graph-Enhanced Retrieval ===================


class TestGraphEnhancedRetrieval:
    @pytest.fixture
    def graph_with_deps(self):
        docs = [
            {"path": "/app/auth.py", "file_name": "auth.py", "content": "from utils import helper"},
            {"path": "/app/utils.py", "file_name": "utils.py", "content": "def helper(): pass"},
            {"path": "/app/users.py", "file_name": "users.py", "content": "x = 1"},
        ]
        g = DependencyGraph()
        g.build(docs)
        return g

    def test_gather_adds_related_chunks(self, graph_with_deps):
        from query import _gather_graph_context

        results = [
            {"path": "/app/auth.py", "file_name": "auth.py", "content": "from utils import helper"}
        ]
        metadata = [
            {"path": "/app/auth.py", "file_name": "auth.py", "content": "..."},
            {"path": "/app/utils.py", "file_name": "utils.py", "content": "def helper(): pass"},
            {"path": "/app/users.py", "file_name": "users.py", "content": "x = 1"},
        ]
        graph_chunks = _gather_graph_context(results, metadata, graph_with_deps)
        assert len(graph_chunks) >= 1
        graph_paths = [c["path"] for c in graph_chunks]
        assert "/app/utils.py" in graph_paths

    def test_graph_chunks_tagged(self, graph_with_deps):
        from query import _gather_graph_context

        results = [{"path": "/app/auth.py", "file_name": "auth.py", "content": "..."}]
        metadata = [
            {"path": "/app/auth.py", "file_name": "auth.py", "content": "..."},
            {"path": "/app/utils.py", "file_name": "utils.py", "content": "..."},
        ]
        graph_chunks = _gather_graph_context(results, metadata, graph_with_deps)
        for chunk in graph_chunks:
            assert chunk.get("_graph") is True

    def test_no_graph_for_isolated_files(self, graph_with_deps):
        from query import _gather_graph_context

        results = [{"path": "/app/users.py", "file_name": "users.py", "content": "x = 1"}]
        metadata = [
            {"path": "/app/auth.py", "file_name": "auth.py", "content": "..."},
            {"path": "/app/users.py", "file_name": "users.py", "content": "x = 1"},
        ]
        graph_chunks = _gather_graph_context(results, metadata, graph_with_deps)
        assert len(graph_chunks) == 0

    def test_no_duplicates_with_search_results(self, graph_with_deps):
        from query import _gather_graph_context

        results = [
            {"path": "/app/auth.py", "file_name": "auth.py", "content": "..."},
            {"path": "/app/utils.py", "file_name": "utils.py", "content": "..."},
        ]
        metadata = [
            {"path": "/app/auth.py", "file_name": "auth.py", "content": "..."},
            {"path": "/app/utils.py", "file_name": "utils.py", "content": "..."},
        ]
        graph_chunks = _gather_graph_context(results, metadata, graph_with_deps)
        assert len(graph_chunks) == 0

    def test_max_two_graph_chunks(self):
        docs = [
            {
                "path": "/a.py",
                "file_name": "a.py",
                "content": "from b import x\nfrom c import y\nfrom d import z",
            },
            {"path": "/b.py", "file_name": "b.py", "content": "x = 1"},
            {"path": "/c.py", "file_name": "c.py", "content": "y = 2"},
            {"path": "/d.py", "file_name": "d.py", "content": "z = 3"},
        ]
        g = DependencyGraph()
        g.build(docs)

        from query import _gather_graph_context

        results = [{"path": "/a.py", "file_name": "a.py", "content": "..."}]
        metadata = [
            {"path": d["path"], "file_name": d["file_name"], "content": d["content"]} for d in docs
        ]
        graph_chunks = _gather_graph_context(results, metadata, g)
        assert len(graph_chunks) <= 2
