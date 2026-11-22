import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

DB_PATH = os.path.join(BASE_DIR, "data", "medmate.db")
SCHEMA_PATH = BASE_DIR / "src" / "medmate" / "db" / "schema.sql"
ROLES_CONFIG_PATH  = BASE_DIR / "config" / "roles.yaml"

BM25_MODEL = os.getenv("BM25_MODEL", "Qdrant/bm25")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
OPENAI_LLM_MODEL = os.getenv("OPENAI_LLM_MODEL", "gpt-4.1-mini")
DATA_DIR = os.path.join(BASE_DIR, "data")
MEDMATE_MAIN_PATH = os.path.join(BASE_DIR, "src", "medmate")

QDRANT_PATH = os.path.join(DATA_DIR, "qdrant_data")

MEMORY_QDRANT_PATH = os.path.join(DATA_DIR, "memory_qdrant_store")
SQLITE_CHECKPOINTER_PATH = os.path.join(DATA_DIR, "checkpointer.db")
SQLITE_EPISODIC_PATH = os.path.join(DATA_DIR, "episodic.db")
PG_URI = os.getenv("PG_URI", "postgresql://medmate:medmate@localhost:5432/medmate")

RAG_IMAGES_PATH = os.path.join(DATA_DIR, "rag_storage", "images")

RAG_QUERIES_PATH = os.path.join(MEDMATE_MAIN_PATH, "evaluation", "golden_dataset", "rag_queries.jsonl")
# GOLDEN_PATH = "evaluation/golden_dataset/rag_queries.jsonl"
RESULTS_CSV_PATH = os.path.join(MEDMATE_MAIN_PATH, "evaluation", "results", "rag_eval_latest.csv")
RAW_RUN_PATH = os.path.join(MEDMATE_MAIN_PATH, "evaluation", "results", "rag_eval_raw.jsonl")

ALLOWED_ORIGINS: list[str] = [
    "http://localhost:3000",   # React dev
    "http://localhost:8501",  # Streamlit dev
]

JWT_SECRET_KEY: str = "change-me-in-env"
JWT_ALGORITHM: str = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
REFRESH_TOKEN_EXPIRE_DAYS: int = 7

# print(RAG_IMAGEA_PATH)
# print(QDRANT_PATH)
# print(DB_PATH)
# print(SCHEMA_PATH)
# print(ROLES_CONFIG_PATH)
# print(RAG_QUERIES_PATH)
# print(RAW_RUN_PATH)