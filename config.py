import os

# PostgreSQL
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "ipc_rag")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")

# Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# LLM
LLM_TEMPERATURE = 0.0

# Chunking - Parent-child strategy
CHILD_CHUNK_SIZE = 256
CHILD_CHUNK_OVERLAP = 50
PARENT_CHUNK_SIZE = 1024
PARENT_CHUNK_OVERLAP = 100
CHUNK_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

# Embedding - BGE-M3 (dense 1024-dim)
EMBEDDING_MODEL = "BAAI/bge-m3"
DENSE_VECTOR_SIZE = 1024

# Retrieval
TOP_K = 20
RERANK_TOP_N = 5
