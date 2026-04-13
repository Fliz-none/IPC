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
CHILD_CHUNK_SIZE = 256
CHILD_CHUNK_OVERLAP = 50
PARENT_CHUNK_SIZE = 1024
PARENT_CHUNK_OVERLAP = 100
CHUNK_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

# Embedding - Gemini embedding (768-dim truncated)
EMBEDDING_MODEL = "gemini-embedding-001"
DENSE_VECTOR_SIZE = 768

# Retrieval
TOP_K = 20
RERANK_TOP_N = 5
