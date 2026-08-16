from pathlib import Path

from pydantic_settings import BaseSettings

BASE_DIR = Path(__file__).resolve().parent


class Settings(BaseSettings):
    ollama_model: str = "llama3"
    ollama_base_url: str = "http://localhost:11434"
    embedding_model: str = "all-MiniLM-L6-v2"
    faiss_index_path: Path = BASE_DIR / "data" / "index.faiss"
    metadata_path: Path = BASE_DIR / "data" / "metadata.pkl"
    db_path: Path = BASE_DIR / "data" / "sessions.db"
    projects_dir: Path = BASE_DIR / "data" / "projects"
    cors_origins: list[str] = ["http://localhost:3000", "http://127.0.0.1:5500"]
    log_level: str = "info"

    model_config = {"env_file": BASE_DIR.parent / ".env", "extra": "ignore"}

    def project_index_path(self, project_id: str) -> Path:
        return self.projects_dir / project_id / "index.faiss"

    def project_metadata_path(self, project_id: str) -> Path:
        return self.projects_dir / project_id / "metadata.pkl"


settings = Settings()
