import json
import logging
import threading
import time
import uuid
from pathlib import Path

from cache import embedding_cache, query_cache
from chunkers import chunk_file
from config import BASE_DIR, settings
from depgraph import DependencyGraph
from evaluate import evaluate_retrieval, report_to_dict
from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from ingest import create_vector_store, load_codebase, load_index, save_index
from pydantic import BaseModel
from query import answer_question, get_embedding, stream_answer
from search import HybridSearcher
from sessions import (
    create_project,
    create_session,
    delete_project,
    delete_session,
    get_messages,
    get_project,
    get_project_by_name,
    get_session,
    init_db,
    list_projects,
    list_sessions,
    update_project_stats,
    update_session_title,
)

logging.basicConfig(
    level=settings.log_level.upper(),
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

init_db()

app = FastAPI(
    title="StackSense",
    description="RAG-based code assistant — indexes your codebase and answers questions using AI",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    request_id = uuid.uuid4().hex[:8]
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start
    response.headers["X-Request-ID"] = request_id
    if request.url.path.startswith("/api"):
        logger.info(
            "%s %s %d %.3fs [%s]",
            request.method,
            request.url.path,
            response.status_code,
            duration,
            request_id,
        )
    return response


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


# --------------- Index Store ---------------


class IndexStore:
    def __init__(self):
        self.index = None
        self.metadata = None
        self.searcher = HybridSearcher()
        self.dep_graph = DependencyGraph()
        self.active_project_id: str | None = None
        self._project_cache: dict[str, tuple] = {}

    def load(self, project_id: str | None = None):
        try:
            self.index, self.metadata = load_index(project_id)
            self.searcher.build(self.index, self.metadata)
            self.active_project_id = project_id
            logger.info("Index loaded: %d vectors (hybrid search ready)", self.index.ntotal)
        except (FileNotFoundError, RuntimeError, OSError):
            if project_id:
                logger.warning("No index found for project %s", project_id)
            else:
                logger.warning("No index found — run ingest first")

    def switch_project(self, project_id: str | None) -> bool:
        if project_id == self.active_project_id:
            return True
        if project_id and project_id in self._project_cache:
            self.index, self.metadata = self._project_cache[project_id]
            self.searcher.build(self.index, self.metadata)
            self.active_project_id = project_id
            return True
        try:
            self.index, self.metadata = load_index(project_id)
            self.searcher.build(self.index, self.metadata)
            if project_id:
                self._project_cache[project_id] = (self.index, self.metadata)
            self.active_project_id = project_id
            return True
        except (FileNotFoundError, RuntimeError, OSError):
            return False

    def set_index(self, index, metadata, project_id: str | None = None):
        self.index = index
        self.metadata = metadata
        self.searcher.build(index, metadata)
        self.active_project_id = project_id
        if project_id:
            self._project_cache[project_id] = (index, metadata)

    def is_ready(self) -> bool:
        return self.index is not None and self.metadata is not None


store = IndexStore()
store.load()


# --------------- Ingestion Jobs ---------------


class IngestionJob:
    def __init__(self, job_id: str, path: str, project_name: str | None = None):
        self.job_id = job_id
        self.path = path
        self.project_name = project_name
        self.status = "pending"
        self.progress: list[dict] = []
        self.result: dict | None = None
        self.error: str | None = None

    def _emit(self, stage: str, detail: str, pct: int):
        self.progress.append({"stage": stage, "detail": detail, "pct": pct, "ts": time.time()})

    def run(self):
        self.status = "running"
        try:
            self._emit("loading", "Scanning files...", 10)
            docs = load_codebase(self.path)
            if not docs:
                self.error = "No supported files found"
                self.status = "failed"
                return

            self._emit("chunking", f"Chunking {len(docs)} files...", 30)
            all_chunks: list[dict] = []
            for doc in docs:
                all_chunks.extend(chunk_file(doc["content"], doc["file_name"], doc["path"]))

            project_id = None
            if self.project_name:
                project = get_project_by_name(self.project_name)
                if not project:
                    project = create_project(self.project_name, self.path)
                project_id = project["id"]
                update_project_stats(project_id, len(docs), len(all_chunks))

            self._emit("embedding", f"Embedding {len(all_chunks)} chunks...", 50)
            index, metadata = create_vector_store(all_chunks)

            self._emit("saving", "Saving index...", 80)
            save_index(index, metadata, project_id)
            store.set_index(index, metadata, project_id)

            self._emit("graph", "Building dependency graph...", 90)
            store.dep_graph.build(docs)
            query_cache.clear()

            self.result = {
                "status": "ok",
                "files_processed": len(docs),
                "chunks_created": len(all_chunks),
                "vectors_stored": index.ntotal,
            }
            if project_id:
                self.result["project_id"] = project_id
                self.result["project_name"] = self.project_name

            self._emit("done", "Ingestion complete!", 100)
            self.status = "completed"
        except Exception as e:
            logger.exception("Ingestion job %s failed", self.job_id)
            self.error = str(e)
            self.status = "failed"
            self._emit("error", str(e), -1)


_jobs: dict[str, IngestionJob] = {}


# --------------- Request / Response Models ---------------


class QueryRequest(BaseModel):
    query: str
    session_id: str | None = None
    project_id: str | None = None


class QueryResponse(BaseModel):
    answer: str


class IngestRequest(BaseModel):
    path: str
    project_name: str | None = None


class SessionCreate(BaseModel):
    title: str = "New Chat"
    project_id: str | None = None


class SessionUpdate(BaseModel):
    title: str


class ProjectCreate(BaseModel):
    name: str
    path: str
    language: str = ""


# --------------- API Router ---------------

api = APIRouter(prefix="/api")


@api.get("/health")
def health():
    return {
        "status": "ok",
        "index_loaded": store.is_ready(),
        "vectors": store.index.ntotal if store.index else 0,
        "hybrid_search": store.searcher.is_ready(),
        "model": settings.ollama_model,
        "embedding_model": settings.embedding_model,
    }


def _ensure_project_loaded(project_id: str | None):
    if (
        project_id
        and project_id != store.active_project_id
        and not store.switch_project(project_id)
    ):
        raise HTTPException(404, "Project index not found. Ingest first.")


@api.post("/ask", response_model=QueryResponse)
def ask(req: QueryRequest):
    _ensure_project_loaded(req.project_id)
    if not store.is_ready():
        raise HTTPException(503, "Index not loaded. Use /api/ingest to build it.")
    answer = answer_question(
        req.query, store.index, store.metadata, req.session_id, store.searcher, store.dep_graph
    )
    return QueryResponse(answer=answer)


@api.post("/stream")
def stream(req: QueryRequest):
    _ensure_project_loaded(req.project_id)
    if not store.is_ready():
        raise HTTPException(503, "Index not loaded. Use /api/ingest to build it.")
    return StreamingResponse(
        stream_answer(
            req.query, store.index, store.metadata, req.session_id, store.searcher, store.dep_graph
        ),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@api.post("/ingest")
def ingest(req: IngestRequest):
    path = Path(req.path).resolve()
    if not path.exists():
        raise HTTPException(400, f"Path does not exist: {path}")
    if not path.is_dir():
        raise HTTPException(400, f"Path is not a directory: {path}")

    docs = load_codebase(str(path))
    if not docs:
        raise HTTPException(400, "No supported files found in the directory")

    all_chunks: list[dict] = []
    for doc in docs:
        all_chunks.extend(chunk_file(doc["content"], doc["file_name"], doc["path"]))

    project_id = None
    if req.project_name:
        project = get_project_by_name(req.project_name)
        if not project:
            project = create_project(req.project_name, str(path))
        project_id = project["id"]
        update_project_stats(project_id, len(docs), len(all_chunks))

    index, metadata = create_vector_store(all_chunks)
    save_index(index, metadata, project_id)
    store.set_index(index, metadata, project_id)
    store.dep_graph.build(docs)
    query_cache.clear()

    result = {
        "status": "ok",
        "files_processed": len(docs),
        "chunks_created": len(all_chunks),
        "vectors_stored": index.ntotal,
    }
    if project_id:
        result["project_id"] = project_id
        result["project_name"] = req.project_name
    return result


# --------------- Background Ingestion ---------------


@api.post("/ingest/start")
def start_ingest(req: IngestRequest):
    path = Path(req.path).resolve()
    if not path.exists():
        raise HTTPException(400, f"Path does not exist: {path}")
    if not path.is_dir():
        raise HTTPException(400, f"Path is not a directory: {path}")

    job_id = uuid.uuid4().hex[:8]
    job = IngestionJob(job_id, str(path), req.project_name)
    _jobs[job_id] = job
    thread = threading.Thread(target=job.run, daemon=True)
    thread.start()
    return {"job_id": job_id, "status": "started"}


@api.get("/ingest/{job_id}/progress")
def ingest_progress(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")

    def stream():
        sent = 0
        while True:
            while sent < len(job.progress):
                event = job.progress[sent]
                yield f"data: {json.dumps(event)}\n\n"
                sent += 1
            if job.status in ("completed", "failed"):
                if job.result:
                    yield f"data: {json.dumps({'stage': 'result', 'result': job.result, 'pct': 100})}\n\n"
                if job.error:
                    yield f"data: {json.dumps({'stage': 'error', 'error': job.error, 'pct': -1})}\n\n"
                break
            time.sleep(0.3)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# --------------- Files & Dependencies ---------------


@api.get("/files")
def list_files():
    if not store.is_ready():
        raise HTTPException(503, "Index not loaded.")
    files: dict[str, dict] = {}
    for m in store.metadata:
        fp = m.get("path", "")
        if fp not in files:
            files[fp] = {
                "path": fp,
                "name": m.get("file_name", ""),
                "chunks": 0,
                "types": [],
            }
        files[fp]["chunks"] += 1
        t = m.get("type", "")
        if t and t not in files[fp]["types"]:
            files[fp]["types"].append(t)
    return sorted(files.values(), key=lambda f: f["path"])


@api.get("/files/{file_path:path}/source")
def get_file_source(file_path: str):
    if not store.metadata or not any(m.get("path") == file_path for m in store.metadata):
        raise HTTPException(403, "File not in indexed codebase")
    candidates = [
        Path(file_path),
        Path(file_path).resolve(),
        BASE_DIR / file_path,
    ]
    actual = None
    for c in candidates:
        if c.is_file():
            actual = c
            break
    if not actual:
        raise HTTPException(404, "File not found")
    try:
        content = actual.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        raise HTTPException(500, f"Cannot read file: {e}") from e
    return {
        "path": str(actual),
        "name": actual.name,
        "content": content,
        "lines": content.count("\n") + 1,
    }


@api.get("/graph")
def get_dependency_graph():
    return store.dep_graph.to_dict()


# --------------- Projects ---------------


@api.post("/projects")
def create_project_endpoint(req: ProjectCreate):
    existing = get_project_by_name(req.name)
    if existing:
        raise HTTPException(409, f"Project '{req.name}' already exists")
    path = Path(req.path).resolve()
    if not path.is_dir():
        raise HTTPException(400, f"Path is not a directory: {path}")
    return create_project(req.name, str(path), req.language)


@api.get("/projects")
def list_projects_endpoint():
    return list_projects()


@api.get("/projects/{project_id}")
def get_project_endpoint(project_id: str):
    p = get_project(project_id)
    if not p:
        raise HTTPException(404, "Project not found")
    return p


@api.delete("/projects/{project_id}")
def delete_project_endpoint(project_id: str):
    if not delete_project(project_id):
        raise HTTPException(404, "Project not found")
    store._project_cache.pop(project_id, None)
    return {"status": "deleted"}


@api.post("/projects/{project_id}/switch")
def switch_project_endpoint(project_id: str):
    if not get_project(project_id):
        raise HTTPException(404, "Project not found")
    if not store.switch_project(project_id):
        raise HTTPException(404, "Project index not found. Ingest first.")
    return {"status": "switched", "project_id": project_id, "vectors": store.index.ntotal}


# --------------- Sessions ---------------


@api.post("/sessions")
def create_session_endpoint(req: SessionCreate):
    return create_session(req.title, req.project_id)


@api.get("/sessions")
def list_sessions_endpoint():
    return list_sessions()


@api.get("/sessions/{session_id}")
def get_session_endpoint(session_id: str):
    s = get_session(session_id)
    if not s:
        raise HTTPException(404, "Session not found")
    return s


@api.delete("/sessions/{session_id}")
def delete_session_endpoint(session_id: str):
    if not delete_session(session_id):
        raise HTTPException(404, "Session not found")
    return {"status": "deleted"}


@api.patch("/sessions/{session_id}")
def update_session_endpoint(session_id: str, req: SessionUpdate):
    if not get_session(session_id):
        raise HTTPException(404, "Session not found")
    update_session_title(session_id, req.title)
    return {"status": "updated"}


@api.get("/sessions/{session_id}/messages")
def get_messages_endpoint(session_id: str):
    if not get_session(session_id):
        raise HTTPException(404, "Session not found")
    return get_messages(session_id)


# --------------- Cache & Evaluation ---------------


@api.get("/cache/stats")
def cache_stats():
    return {
        "embedding_cache": embedding_cache.stats(),
        "query_cache": query_cache.stats(),
    }


@api.get("/eval")
def eval_endpoint():
    if not store.is_ready():
        raise HTTPException(503, "Index not loaded. Use /api/ingest first.")
    report = evaluate_retrieval(store.index, store.metadata, get_embedding, store.searcher)
    return report_to_dict(report)


# --------------- Mount ---------------

app.include_router(api)

frontend_dir = BASE_DIR.parent / "frontend"
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")
