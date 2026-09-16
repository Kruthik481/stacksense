import json
import logging
from collections.abc import Generator

import faiss
import llm
import numpy as np
from cache import embedding_cache
from depgraph import DependencyGraph
from embeddings import embed
from search import HybridSearcher
from sessions import add_message, auto_title, get_messages

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an expert code assistant.

You MUST ONLY answer using the provided code context.
Do NOT explain generally.
Do NOT guess.
Do NOT add outside knowledge.

If the answer is not directly in the code, say:
"I could not find this in the codebase."

When possible, quote or reference the exact code.
Format your answers with markdown: use ```language code blocks, **bold**, bullet points, and headers.
Keep answers concise but thorough."""


def get_embedding(text: str) -> np.ndarray:
    cached = embedding_cache.get(text)
    if cached is not None:
        return cached
    emb = embed(text)
    embedding_cache.put(text, emb)
    return emb


def rewrite_query(query: str, history: list[dict]) -> str:
    query = query.strip()
    if not history:
        return query

    last_user_msgs = [m["content"] for m in history if m["role"] == "user"]
    if not last_user_msgs:
        return query

    last_question = None
    for msg in reversed(last_user_msgs):
        if len(msg.split()) > 3:
            last_question = msg
            break
    if last_question is None:
        last_question = last_user_msgs[-1]

    if len(query.split()) < 3:
        return f"{last_question} {query}"
    return query


def search(query: str, index: faiss.Index, metadata: list[dict], k: int = 20) -> list[dict]:
    vec = np.array([get_embedding(query)]).astype("float32")
    distances, indices = index.search(vec, k)

    results = []
    for dist, idx in zip(distances[0], indices[0], strict=False):
        if idx == -1:
            continue
        if dist < 1.5:
            results.append(metadata[idx])
    return results[:5]


def rerank_results(query: str, results: list[dict]) -> list[dict]:
    words = set(query.lower().split())
    scored = []
    for r in results:
        content = r["content"].lower()
        score = sum(1 for w in words if w in content)
        scored.append((score, r))
    scored.sort(reverse=True, key=lambda x: x[0])
    return [r for _, r in scored]


def _build_context(results: list[dict]) -> str:
    parts = []
    for r in results:
        header = f"FILE: {r['file_name']}"
        if r.get("start_line"):
            header += f" (lines {r['start_line']}-{r.get('end_line', '?')})"
        if r.get("name") and r["name"] != r["file_name"]:
            header += f" | {r.get('type', 'chunk')}: {r['name']}"
        parts.append(f"--- {header} ---\n{r['content'][:1000]}")
    return "\n\n---\n\n".join(parts)


def _build_messages(query: str, context: str, history: list[dict]) -> list[dict]:
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for msg in history[-6:]:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append(
        {
            "role": "user",
            "content": (
                f"Use ONLY the provided code context to answer.\n\n"
                f"Context:\n{context}\n\n"
                f"Question: {query}"
            ),
        }
    )
    return messages


def _gather_graph_context(
    results: list[dict],
    metadata: list[dict],
    dep_graph: DependencyGraph,
) -> list[dict]:
    hit_files = {r.get("path", "") for r in results}
    related_paths: set[str] = set()
    for path in hit_files:
        related_paths |= dep_graph.get_related_files(path, depth=1)
    related_paths -= hit_files

    if not related_paths:
        return []

    graph_chunks: list[dict] = []
    seen: set[str] = set()
    for m in metadata:
        mp = m.get("path", "")
        if mp in related_paths and mp not in seen:
            chunk = dict(m)
            chunk["_graph"] = True
            graph_chunks.append(chunk)
            seen.add(mp)
            if len(graph_chunks) >= 2:
                break
    return graph_chunks


def _retrieve(
    query: str,
    index: faiss.Index,
    metadata: list[dict],
    history: list[dict],
    searcher: HybridSearcher | None = None,
    dep_graph: DependencyGraph | None = None,
) -> tuple[str, list[dict]]:
    rewritten = rewrite_query(query, history)
    logger.debug("Original: %s | Rewritten: %s", query, rewritten)

    if searcher and searcher.is_ready():
        embedding = get_embedding(rewritten)
        results = searcher.search(rewritten, embedding, k=20, final_k=5)
        results = rerank_results(query, results)[:3]
        logger.debug("Hybrid search returned %d results", len(results))
    else:
        results = search(rewritten, index, metadata)
        results = rerank_results(query, results)[:3]

    if dep_graph and dep_graph.files:
        graph_chunks = _gather_graph_context(results, metadata, dep_graph)
        if graph_chunks:
            logger.debug("Graph added %d related chunks", len(graph_chunks))
            results = results + graph_chunks

    context = _build_context(results)
    return context, results


def answer_question(
    query: str,
    index: faiss.Index,
    metadata: list[dict],
    session_id: str | None = None,
    searcher: HybridSearcher | None = None,
    dep_graph: DependencyGraph | None = None,
) -> str:
    history = get_messages(session_id, limit=10) if session_id else []
    context, _ = _retrieve(query, index, metadata, history, searcher, dep_graph)
    messages = _build_messages(query, context, history)

    answer = llm.chat(messages)

    if session_id:
        add_message(session_id, "user", query)
        add_message(session_id, "assistant", answer)
        if not history:
            auto_title(session_id, query)

    return answer


def stream_answer(
    query: str,
    index: faiss.Index,
    metadata: list[dict],
    session_id: str | None = None,
    searcher: HybridSearcher | None = None,
    dep_graph: DependencyGraph | None = None,
) -> Generator[str, None, None]:
    history = get_messages(session_id, limit=10) if session_id else []
    context, results = _retrieve(query, index, metadata, history, searcher, dep_graph)
    messages = _build_messages(query, context, history)

    sources = []
    for r in results:
        src = {
            "file": r["file_name"],
            "lines": f"{r.get('start_line', '?')}-{r.get('end_line', '?')}",
            "name": r.get("name", r["file_name"]),
        }
        if r.get("_graph"):
            src["via"] = "dependency graph"
        sources.append(src)
    yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"

    full_answer = ""
    try:
        for token in llm.stream_chat(messages):
            full_answer += token
            yield f"data: {json.dumps({'type': 'token', 'token': token})}\n\n"
    except llm.LLMError as e:
        # Headers are already sent, so report the failure in-band and skip saving history.
        yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
        return

    yield f"data: {json.dumps({'type': 'done'})}\n\n"

    if session_id:
        add_message(session_id, "user", query)
        add_message(session_id, "assistant", full_answer)
        if not history:
            auto_title(session_id, query)
