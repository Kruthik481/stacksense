import pytest
from chunkers import _infer_name, chunk_file


class TestPythonChunking:
    def test_single_function(self):
        code = "def greet(name):\n    return f'Hello {name}'\n"
        chunks = chunk_file(code, "app.py", "/app.py")
        assert len(chunks) >= 1
        func_chunks = [c for c in chunks if c["type"] == "FunctionDef"]
        assert len(func_chunks) == 1
        assert func_chunks[0]["name"] == "greet"
        assert func_chunks[0]["start_line"] == 1

    def test_class_with_methods(self):
        code = (
            "class Calculator:\n"
            "    def add(self, a, b):\n"
            "        return a + b\n"
            "    def subtract(self, a, b):\n"
            "        return a - b\n"
        )
        chunks = chunk_file(code, "calc.py", "/calc.py")
        names = {c["name"] for c in chunks}
        assert "Calculator" in names
        assert "Calculator.add" in names
        assert "Calculator.subtract" in names

    def test_async_function(self):
        code = "async def fetch_data(url):\n    response = await client.get(url)\n    return response.json()\n"
        chunks = chunk_file(code, "api.py", "/api.py")
        assert any(c["type"] == "AsyncFunctionDef" for c in chunks)

    def test_decorated_function(self):
        code = "@app.get('/health')\ndef health():\n    return {'status': 'ok'}\n"
        chunks = chunk_file(code, "routes.py", "/routes.py")
        func = next(c for c in chunks if c["type"] == "FunctionDef")
        assert func["start_line"] == 1
        assert "@app.get" in func["content"]

    def test_syntax_error_fallback(self):
        code = "def broken(\n    this is not valid python at all\n    more garbage here !!!\n"
        chunks = chunk_file(code, "bad.py", "/bad.py")
        assert len(chunks) >= 1
        assert chunks[0]["type"] in ("chunk", "file")

    def test_small_content_skipped(self):
        code = "x = 1"
        chunks = chunk_file(code, "tiny.py", "/tiny.py")
        assert len(chunks) == 0


class TestJavaScriptChunking:
    def test_function_split(self):
        code = (
            "const x = 1;\nconsole.log(x);\n\n"
            "function greet(name) {\n  return `Hello ${name}`;\n}\n\n"
            "const add = (a, b) => a + b;\n"
        )
        chunks = chunk_file(code, "app.js", "/app.js")
        assert len(chunks) >= 1

    def test_export_detection(self):
        code = "export default function handler(req, res) {\n  res.send('ok');\n}\n"
        chunks = chunk_file(code, "handler.ts", "/handler.ts")
        assert len(chunks) >= 1


class TestGenericChunking:
    def test_whole_file_fallback(self):
        code = "<html>\n<head><title>Test</title></head>\n<body><p>Hello world paragraph content here for testing</p></body>\n</html>"
        chunks = chunk_file(code, "page.html", "/page.html")
        assert len(chunks) == 1
        assert chunks[0]["type"] in ("file", "chunk")

    def test_empty_content(self):
        chunks = chunk_file("", "empty.py", "/empty.py")
        assert len(chunks) == 0


class TestInferName:
    @pytest.mark.parametrize(
        "line,expected",
        [
            ("def hello():", "hello"),
            ("class MyClass:", "MyClass"),
            ("function fetchData() {", "fetchData"),
            ("const API_URL = 'http://...'", "API_URL"),
            ("export default function handler()", "handler"),
        ],
    )
    def test_names(self, line, expected):
        assert _infer_name(line) == expected
