"""
Centralized application configuration.

All tunables live here and are overridable via environment variables / .env,
so the same codebase can run with different LLM providers, chunk sizes, or
storage locations without touching application code.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
STORAGE_DIR = BASE_DIR / "storage"
UPLOAD_DIR = STORAGE_DIR / "uploads"
VECTOR_INDEX_DIR = STORAGE_DIR / "vector_index"
DB_PATH = STORAGE_DIR / "study_assistant.db"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_INDEX_DIR.mkdir(parents=True, exist_ok=True)


class Settings:
    # --- Paths ---
    STORAGE_DIR = STORAGE_DIR
    UPLOAD_DIR = UPLOAD_DIR
    VECTOR_INDEX_DIR = VECTOR_INDEX_DIR

    # --- Database ---
    DATABASE_URL: str = f"sqlite:///{DB_PATH}"

    # --- LLM provider ---
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "anthropic")

    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    ANTHROPIC_MODEL: str = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")

    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    EMBEDDING_DIM: int = 384

    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "800"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "150"))
    TOP_K_RETRIEVAL: int = int(os.getenv("TOP_K_RETRIEVAL", "4"))

    MAX_UPLOAD_MB: int = int(os.getenv("MAX_UPLOAD_MB", "20"))
    CORS_ORIGINS: list = ["*"]

settings = Settings()
