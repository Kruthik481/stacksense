import importlib.util
from pathlib import Path

from fastapi import FastAPI

ENTRYPOINT = Path(__file__).resolve().parent.parent / "app.py"


def test_root_entrypoint_exposes_the_backend_app():
    spec = importlib.util.spec_from_file_location("vercel_entrypoint", ENTRYPOINT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    from main import app

    assert isinstance(module.app, FastAPI)
    assert module.app is app
