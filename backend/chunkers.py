import ast
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)


def chunk_file(content: str, file_name: str, path: str) -> list[dict]:
    ext = Path(file_name).suffix.lower()

    if ext == ".py":
        return _chunk_python(content, file_name, path)
    if ext in {".js", ".ts", ".jsx", ".tsx"}:
        return _chunk_javascript(content, file_name, path)
    if ext in {".java", ".cpp", ".c", ".h", ".go", ".rs"}:
        return _chunk_c_style(content, file_name, path)
    return _chunk_generic(content, file_name, path)


# ---------- Python (AST-based) ----------


def _chunk_python(content: str, file_name: str, path: str) -> list[dict]:
    try:
        tree = ast.parse(content)
    except SyntaxError:
        logger.debug("AST parse failed for %s, falling back to regex", file_name)
        return _chunk_generic(content, file_name, path)

    lines = content.split("\n")
    chunks: list[dict] = []
    covered: set[int] = set()

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            chunk = _extract_ast_node(node, lines, file_name, path)
            if chunk:
                chunks.append(chunk)
                covered.update(range(chunk["start_line"], chunk["end_line"] + 1))

        elif isinstance(node, ast.ClassDef):
            chunk = _extract_ast_node(node, lines, file_name, path)
            if chunk:
                chunks.append(chunk)
                covered.update(range(chunk["start_line"], chunk["end_line"] + 1))

            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    mc = _extract_ast_node(child, lines, file_name, path)
                    if mc:
                        mc["name"] = f"{node.name}.{child.name}"
                        chunks.append(mc)

    module_lines = [
        line
        for i, line in enumerate(lines, 1)
        if i not in covered and line.strip() and not line.strip().startswith("#")
    ]
    if module_lines and len("\n".join(module_lines)) > 30:
        chunks.append(
            {
                "content": "\n".join(module_lines),
                "file_name": file_name,
                "path": path,
                "type": "module",
                "name": file_name,
                "start_line": 1,
                "end_line": len(lines),
            }
        )

    if not chunks and len(content.strip()) > 30:
        chunks.append(_whole_file(content, file_name, path))

    return chunks


def _extract_ast_node(
    node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
    lines: list[str],
    file_name: str,
    path: str,
) -> dict | None:
    start = node.lineno
    if node.decorator_list:
        start = node.decorator_list[0].lineno
    end = node.end_lineno
    text = "\n".join(lines[start - 1 : end])

    if len(text.strip()) < 30:
        return None

    return {
        "content": text,
        "file_name": file_name,
        "path": path,
        "type": type(node).__name__,
        "name": node.name,
        "start_line": start,
        "end_line": end,
    }


# ---------- JavaScript / TypeScript ----------


def _chunk_javascript(content: str, file_name: str, path: str) -> list[dict]:
    pattern = (
        r"\n(?=(?:export\s+)?(?:default\s+)?(?:async\s+)?"
        r"(?:function|class|const|let|var)\s)"
    )
    return _split_by_pattern(content, pattern, file_name, path)


# ---------- C-style languages ----------


def _chunk_c_style(content: str, file_name: str, path: str) -> list[dict]:
    pattern = (
        r"\n(?=(?:public|private|protected|static|virtual|async)?\s*"
        r"(?:class|struct|enum|interface|func|fn)\s|\w+[\s*&]+\w+\s*\()"
    )
    return _split_by_pattern(content, pattern, file_name, path)


# ---------- Generic (regex) ----------


def _chunk_generic(content: str, file_name: str, path: str) -> list[dict]:
    pattern = r"\n(?=def |class |function |const |export |module\.exports)"
    return _split_by_pattern(content, pattern, file_name, path)


# ---------- Helpers ----------


def _split_by_pattern(content: str, pattern: str, file_name: str, path: str) -> list[dict]:
    parts = re.split(pattern, content)
    chunks: list[dict] = []
    line_offset = 1

    for part in parts:
        stripped = part.strip()
        if len(stripped) > 50:
            chunks.append(
                {
                    "content": stripped,
                    "file_name": file_name,
                    "path": path,
                    "type": "chunk",
                    "name": _infer_name(stripped),
                    "start_line": line_offset,
                    "end_line": line_offset + part.count("\n"),
                }
            )
        line_offset += part.count("\n") + 1

    if not chunks and len(content.strip()) > 50:
        chunks.append(_whole_file(content, file_name, path))

    return chunks


def _whole_file(content: str, file_name: str, path: str) -> dict:
    return {
        "content": content,
        "file_name": file_name,
        "path": path,
        "type": "file",
        "name": file_name,
        "start_line": 1,
        "end_line": content.count("\n") + 1,
    }


def _infer_name(chunk: str) -> str:
    first_line = chunk.split("\n")[0].strip()
    for pat in [
        r"(?:def|function|class|struct|enum|interface)\s+(\w+)",
        r"(?:const|let|var)\s+(\w+)",
        r"export\s+(?:default\s+)?(?:function|class|const)\s+(\w+)",
    ]:
        m = re.search(pat, first_line)
        if m:
            return m.group(1)
    return first_line[:40]
