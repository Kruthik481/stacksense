"""Vercel entrypoint. Backend modules import each other by bare name, so expose backend/ first."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "backend"))

from main import app

__all__ = ["app"]
