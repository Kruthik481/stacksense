<div align="center">

# StackSense

**AI-powered code assistant that indexes your codebase and answers questions using RAG**

Built with FAISS · BM25 · Ollama · FastAPI · sentence-transformers

[![CI](https://github.com/Kruthik481/stacksense/actions/workflows/ci.yml/badge.svg)](https://github.com/Kruthik481/stacksense/actions)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-152%20passing-brightgreen.svg)](#testing)

</div>

---

StackSense indexes any codebase, builds a semantic search index, and answers natural language questions about your code — grounding every answer in actual source files. It combines vector similarity search (FAISS) with keyword matching (BM25) using Reciprocal Rank Fusion, and enhances context with a code dependency graph so the LLM sees not just the relevant file, but its imports and dependents too.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Frontend (SPA)                           │
│  ┌──────────┐  ┌──────────────┐  ┌────────────┐  ┌──────────┐  │
│  │   Chat   │  │  Dashboard   │  │File Browser│  │  Theme   │  │
│  │ (SSE     │  │ (Cache stats,│  │(Tree view, │  │  Toggle  │  │
│  │ streaming)│  │  dep graph)  │  │ source)    │  │          │  │
│  └──────────┘  └──────────────┘  └────────────┘  └──────────┘  │
└────────────────────────────┬────────────────────────────────────┘
                             │ HTTP / SSE
┌────────────────────────────▼────────────────────────────────────┐
│                     FastAPI Backend                              │
│                                                                  │
│  ┌─────────────────── Query Pipeline ───────────────────────┐   │
│  │                                                           │   │
│  │  Query → Rewrite → ┌─────────────┐ → RRF → Rerank       │   │
│  │                     │ FAISS (vec) │   Fusion              │   │
│  │                     │ BM25 (kw)   │                       │   │
│  │                     └─────────────┘                       │   │
│  │                           │                               │   │
│  │              ┌────────────▼──────────────┐                │   │
│  │              │   Dependency Graph         │                │   │
│  │              │   (inject related files)   │                │   │
│  │              └────────────┬──────────────┘                │   │
│  │                           │                               │   │
│  │              ┌────────────▼──────────────┐                │   │
│  │              │   LLM Context Builder      │                │   │
│  │              │   (search + graph chunks)  │                │   │
│  │              └────────────┬──────────────┘                │   │
│  └───────────────────────────┼───────────────────────────────┘   │
│                              │                                    │
│  ┌───────────┐  ┌────────────▼───┐  ┌──────────┐  ┌──────────┐  │
│  │ Embedding │  │  Ollama LLM    │  │ Session  │  │  Cache   │  │
│  │ Cache     │  │  (llama3)      │  │ Store    │  │  (LRU+   │  │
│  │ (LRU)     │  │                │  │ (SQLite) │  │   TTL)   │  │
│  └───────────┘  └────────────────┘  └──────────┘  └──────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

## Key Features

### RAG Pipeline
- **Hybrid Search** — FAISS vector similarity + BM25 keyword scoring, fused with Reciprocal Rank Fusion (RRF) for better recall than either method alone
- **Graph-Enhanced Context** — dependency graph automatically injects related files (imports/dependents) into the LLM context, so answers consider the full picture
- **AST-Based Chunking** — Python files are split by function/class using the AST; JS/TS use regex-based splitting; other languages get intelligent size-based chunking
- **Query Rewriting** — follow-up questions are rewritten using conversation history for better retrieval on short queries like "what about that?"
- **Streaming Responses** — token-by-token SSE streaming with source attribution chips

### Performance & Caching
- **Embedding Cache** — LRU cache (2048 entries) avoids re-encoding repeated text, with thread-safe access
- **Query Cache** — TTL-based cache (300s) for search results, scoped per project
- **Background Ingestion** — long-running indexing jobs run in background threads with real-time SSE progress updates

### Multi-Project Support
- **Project Isolation** — each project gets its own FAISS index, metadata, and dependency graph
- **Hot Switching** — switch between project indices without server restart, with in-memory project cache
- **Session Management** — chat sessions with message history, auto-generated titles, per-project scoping

### Frontend
- **Chat Interface** — real-time streaming with markdown rendering, syntax highlighting, code block copy
- **Analytics Dashboard** — cache hit rates (ring charts), index stats, dependency graph metrics, most-connected files bar chart
- **File Browser** — collapsible tree view of indexed files with source code preview and line numbers
- **Dark/Light Theme** — toggle with localStorage persistence

### Developer Experience
- **152 Tests** across 8 test files — unit tests for every module, integration tests for all API endpoints
- **CI/CD Pipeline** — GitHub Actions with lint (ruff), type checking (mypy), tests (pytest), Docker build
- **Docker Compose** — one-command deployment with Ollama + app services, health checks, persistent volumes
- **Retrieval Evaluation** — built-in harness measuring precision@k, MRR, and query latency

## Tech Stack

| Layer | Technology |
|-------|-----------|
| LLM | Ollama (llama3) |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2, 384d) |
| Vector Search | FAISS (IndexFlatL2) |
| Keyword Search | Custom BM25 implementation |
| Framework | FastAPI with APIRouter |
| Database | SQLite (sessions, projects, messages) |
| Frontend | Vanilla HTML/CSS/JS (single file SPA) |
| Containerization | Docker multi-stage build, docker-compose |
| CI/CD | GitHub Actions (4-stage pipeline) |
| Linting | Ruff |
| Testing | Pytest (152 tests) |

## Quick Start

### Prerequisites

- Python 3.12+
- [Ollama](https://ollama.ai) installed and running

### 1. Clone and install

```bash
git clone https://github.com/Kruthik481/stacksense.git
cd stacksense
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Pull the LLM model

```bash
ollama pull llama3
```

### 3. Index a codebase

```bash
cd backend
python ingest.py /path/to/your/codebase
```

### 4. Start the server

```bash
uvicorn main:app --reload --app-dir backend
```

Open [http://localhost:8000](http://localhost:8000) and start asking questions about your code.

### Docker (alternative)

```bash
docker compose up --build
```

This starts Ollama + StackSense with persistent volumes. The app is available at `http://localhost:8000`.

## Configuration

Copy `.env.example` to `.env` and customize:

```bash
cp .env.example .env
```

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_MODEL` | `llama3` | LLM model name |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API endpoint |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence transformer model |
| `LOG_LEVEL` | `info` | Logging level |

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/ask` | Ask a question (blocking response) |
| `POST` | `/api/stream` | Ask a question (SSE streaming) |
| `POST` | `/api/ingest` | Index a codebase directory |
| `POST` | `/api/ingest/start` | Start background indexing job |
| `GET` | `/api/ingest/{id}/progress` | SSE stream of indexing progress |
| `GET` | `/api/health` | Index status, model info |
| `GET` | `/api/files` | List all indexed files |
| `GET` | `/api/files/{path}/source` | Get source code of indexed file |
| `GET` | `/api/graph` | Dependency graph (nodes + edges) |
| `GET` | `/api/cache/stats` | Cache hit rates and sizes |
| `GET` | `/api/eval` | Run retrieval evaluation |
| `POST` | `/api/projects` | Create a project |
| `GET` | `/api/projects` | List projects |
| `POST` | `/api/sessions` | Create a chat session |
| `GET` | `/api/sessions` | List chat sessions |
| `GET` | `/api/sessions/{id}/messages` | Get session messages |

## Testing

```bash
pip install -r requirements-dev.txt
cd backend
pytest ../tests/ -v
```

```
152 passed in ~15s
```

Test coverage includes:
- **Cache** — LRU eviction, TTL expiry, thread safety, stats
- **Dependency Graph** — Python/JS/Go import parsing, graph traversal, related files
- **Graph-Enhanced Retrieval** — context injection, deduplication, isolation
- **Hybrid Search** — BM25 scoring, RRF fusion, keyword boosting
- **Chunkers** — AST splitting, regex fallback, edge cases
- **Evaluation** — precision@k, MRR, latency recording
- **API** — all 16 endpoints, error cases, streaming

## Project Structure

```
stacksense/
├── backend/
│   ├── main.py          # FastAPI app, routes, IndexStore, IngestionJob
│   ├── query.py         # RAG pipeline — retrieve, rerank, graph context, stream
│   ├── search.py        # BM25 + FAISS hybrid search with RRF
│   ├── ingest.py        # Codebase loading, FAISS index creation
│   ├── chunkers.py      # AST-based Python chunking, JS/generic splitting
│   ├── depgraph.py      # Code dependency graph (import parsing)
│   ├── cache.py         # LRU embedding cache + TTL query cache
│   ├── evaluate.py      # Retrieval evaluation (precision@k, MRR)
│   ├── sessions.py      # SQLite session/project/message management
│   └── config.py        # pydantic-settings configuration
├── frontend/
│   └── index.html       # Single-file SPA (chat, dashboard, file browser)
├── tests/               # 152 tests across 8 files
├── .github/workflows/   # CI/CD pipeline
├── Dockerfile           # Multi-stage Python build
├── docker-compose.yml   # Ollama + app orchestration
└── requirements.txt     # Production dependencies
```

## How It Works

1. **Ingest** — Walk the codebase, read supported files, split into semantic chunks (functions, classes, or sized blocks), embed with sentence-transformers, store in a FAISS index

2. **Search** — On a query, run both FAISS vector search and BM25 keyword search in parallel, merge with Reciprocal Rank Fusion, rerank by keyword overlap

3. **Graph Enhance** — Look up which files the search hits import from or are imported by (via AST-parsed dependency graph), inject up to 2 related chunks into context

4. **Generate** — Build a prompt with the code context + conversation history, stream the response token-by-token from Ollama, save to session history

## License

MIT
