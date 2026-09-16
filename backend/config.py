import os
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings

BASE_DIR = Path(__file__).resolve().parent


def _on_vercel() -> bool:
    return bool(os.environ.get("VERCEL"))


def _default_db_path() -> Path:
    # Vercel Functions have a read-only filesystem except /tmp.
    if _on_vercel():
        return Path("/tmp/stacksense/sessions.db")
    return BASE_DIR / "data" / "sessions.db"


class Settings(BaseSettings):
    llm_provider: Literal["ollama", "groq"] = "ollama"
    ollama_model: str = "llama3"
    ollama_base_url: str = "http://localhost:11434"
    groq_api_key: SecretStr | None = None
    groq_model: str = "llama-3.1-8b-instant"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    # Both live outside backend/ so the demo index never ingests the model's JSON files.
    # Local runs download into the cache; deploy builds export a flat copy into the bundle
    # dir, which wins when present (the cache's symlinks would ship the weights twice).
    embedding_cache_dir: Path = BASE_DIR.parent / "models" / "cache"
    embedding_bundle_dir: Path = BASE_DIR.parent / "models" / "bundled"
    faiss_index_path: Path = BASE_DIR / "data" / "index.faiss"
    metadata_path: Path = BASE_DIR / "data" / "metadata.pkl"
    db_path: Path = Field(default_factory=_default_db_path)
    projects_dir: Path = BASE_DIR / "data" / "projects"
    cors_origins: list[str] = ["http://localhost:3000", "http://127.0.0.1:5500"]
    log_level: str = "info"
    # Public deployments: block endpoints that read or change the server filesystem,
    # and rate-limit LLM calls per client. Defaults on for Vercel so a missing env var
    # can never expose ingestion on the live site.
    public_demo: bool = Field(default_factory=_on_vercel)
    rate_limit_per_minute: int = 10

    model_config = {"env_file": BASE_DIR.parent / ".env", "extra": "ignore"}

    def project_index_path(self, project_id: str) -> Path:
        return self.projects_dir / project_id / "index.faiss"

    def project_metadata_path(self, project_id: str) -> Path:
        return self.projects_dir / project_id / "metadata.pkl"


settings = Settings()
