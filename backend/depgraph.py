"""Code dependency graph — parses imports to build a directed graph of file relationships."""

import ast
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)


def parse_python_imports(content: str) -> list[str]:
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return _regex_python_imports(content)

    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)
    return imports


def _regex_python_imports(content: str) -> list[str]:
    imports = []
    for m in re.finditer(r"^\s*(?:from|import)\s+([\w.]+)", content, re.MULTILINE):
        imports.append(m.group(1))
    return imports


def parse_js_imports(content: str) -> list[str]:
    imports = []
    for m in re.finditer(
        r"""(?:import\s+.*?\s+from\s+|import\s*\(|require\s*\()\s*['"]([^'"]+)['"]""",
        content,
    ):
        imports.append(m.group(1))
    for m in re.finditer(r"""export\s+.*?\s+from\s+['"]([^'"]+)['"]""", content):
        imports.append(m.group(1))
    return imports


def parse_go_imports(content: str) -> list[str]:
    imports = []
    for m in re.finditer(r'"([^"]+)"', content):
        val = m.group(1)
        if "/" in val or val in ("fmt", "os", "io", "net", "log", "sync", "math", "sort"):
            imports.append(val)
    return imports


def parse_imports(content: str, file_name: str) -> list[str]:
    ext = Path(file_name).suffix.lower()
    if ext == ".py":
        return parse_python_imports(content)
    if ext in {".js", ".ts", ".jsx", ".tsx"}:
        return parse_js_imports(content)
    if ext == ".go":
        return parse_go_imports(content)
    return []


def _resolve_module(module: str, files: dict[str, str]) -> str | None:
    candidates = [
        module.replace(".", "/") + ".py",
        module.replace(".", "/") + "/__init__.py",
        module.split(".")[-1] + ".py",
    ]
    for c in candidates:
        for file_path in files:
            if file_path.endswith(c) or Path(file_path).stem == module.split(".")[-1]:
                return file_path
    return None


def _resolve_js_import(imp: str, source_file: str, files: dict[str, str]) -> str | None:
    if imp.startswith("."):
        source_dir = str(Path(source_file).parent)
        base = str((Path(source_dir) / imp).resolve()) if source_dir != "." else imp
    else:
        base = imp

    for ext in ["", ".js", ".ts", ".jsx", ".tsx", "/index.js", "/index.ts"]:
        candidate = base + ext
        for file_path in files:
            if file_path.endswith(candidate) or Path(file_path).stem == Path(candidate).stem:
                return file_path
    return None


class DependencyGraph:
    def __init__(self):
        self.edges: dict[str, set[str]] = {}
        self.reverse_edges: dict[str, set[str]] = {}
        self.file_imports: dict[str, list[str]] = {}
        self.files: set[str] = set()

    def build(self, documents: list[dict]) -> "DependencyGraph":
        file_contents = {d["path"]: d["content"] for d in documents}
        file_names = {d["path"]: d["file_name"] for d in documents}
        self.files = set(file_contents.keys())

        for file_path, content in file_contents.items():
            file_name = file_names[file_path]
            raw_imports = parse_imports(content, file_name)
            self.file_imports[file_path] = raw_imports

            resolved = set()
            ext = Path(file_name).suffix.lower()
            for imp in raw_imports:
                if ext == ".py":
                    target = _resolve_module(imp, file_contents)
                elif ext in {".js", ".ts", ".jsx", ".tsx"}:
                    target = _resolve_js_import(imp, file_path, file_contents)
                else:
                    target = None

                if target and target != file_path:
                    resolved.add(target)

            self.edges[file_path] = resolved
            for target in resolved:
                if target not in self.reverse_edges:
                    self.reverse_edges[target] = set()
                self.reverse_edges[target].add(file_path)

        logger.info(
            "Dependency graph built: %d files, %d edges",
            len(self.files),
            sum(len(v) for v in self.edges.values()),
        )
        return self

    def get_dependencies(self, file_path: str) -> list[str]:
        return sorted(self.edges.get(file_path, set()))

    def get_dependents(self, file_path: str) -> list[str]:
        return sorted(self.reverse_edges.get(file_path, set()))

    def get_related_files(self, file_path: str, depth: int = 1) -> set[str]:
        related: set[str] = set()
        frontier = {file_path}
        for _ in range(depth):
            next_frontier = set()
            for f in frontier:
                deps = self.edges.get(f, set())
                rev = self.reverse_edges.get(f, set())
                next_frontier |= deps | rev
            next_frontier -= related
            next_frontier.discard(file_path)
            related |= next_frontier
            frontier = next_frontier
        return related

    def to_dict(self) -> dict:
        nodes = []
        for f in sorted(self.files):
            nodes.append(
                {
                    "path": f,
                    "name": Path(f).name,
                    "imports": self.file_imports.get(f, []),
                    "dependencies": self.get_dependencies(f),
                    "dependents": self.get_dependents(f),
                }
            )
        return {
            "nodes": nodes,
            "total_files": len(self.files),
            "total_edges": sum(len(v) for v in self.edges.values()),
        }
