import os

# Supabase PostgreSQL
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/postgres"
)

# Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# Cohere (reranking)
COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")

# LLM
LLM_TEMPERATURE = 0.0

# Chunking - Parent-child strategy
CHILD_CHUNK_SIZE = 256
CHILD_CHUNK_OVERLAP = 50
PARENT_CHUNK_SIZE = 1024
PARENT_CHUNK_OVERLAP = 100
CHUNK_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

# Embedding - Gemini text-embedding-004 (768-dim)
EMBEDDING_MODEL = "gemini-embedding-001"
DENSE_VECTOR_SIZE = 768

# Retrieval
TOP_K = 20
RERANK_TOP_N = 5
