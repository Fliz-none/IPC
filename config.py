import os

def _get_secret(key: str, default: str = "") -> str:
    """Read from Streamlit secrets first, then env vars."""
    try:
        import streamlit as st
        return st.secrets.get(key, os.getenv(key, default))
    except Exception:
        return os.getenv(key, default)

# Supabase PostgreSQL
DATABASE_URL = _get_secret(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/postgres"
)

# Gemini
GEMINI_API_KEY = _get_secret("GEMINI_API_KEY", "")
GEMINI_MODEL = _get_secret("GEMINI_MODEL", "gemini-2.5-flash")

# Cohere (reranking)
COHERE_API_KEY = _get_secret("COHERE_API_KEY", "")

# LLM
LLM_TEMPERATURE = 0.0

# Chunking - Parent-child strategy
CHILD_CHUNK_SIZE = 512
CHILD_CHUNK_OVERLAP = 100
PARENT_CHUNK_SIZE = 2048
PARENT_CHUNK_OVERLAP = 200
CHUNK_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

# Embedding - Cohere embed-multilingual-v3.0 (1024-dim)
EMBEDDING_MODEL = "embed-multilingual-v3.0"
DENSE_VECTOR_SIZE = 1024

# Retrieval
TOP_K = 20
RERANK_TOP_N = 5
