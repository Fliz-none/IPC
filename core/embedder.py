"""Embedding via Cohere embed-multilingual-v3.0 API with auto-retry."""

import time
import os

import cohere

EMBED_BATCH_SIZE = 96  # Cohere supports up to 96 texts per request
MAX_RETRIES = 5
EMBEDDING_MODEL = "embed-multilingual-v3.0"

_client = None


def _get_client(api_key: str = ""):
    global _client
    key = api_key
    if not key:
        try:
            import streamlit as st
            if hasattr(st, "secrets") and "COHERE_API_KEY" in st.secrets:
                key = st.secrets["COHERE_API_KEY"]
        except Exception:
            pass
    if not key:
        key = os.environ.get("COHERE_API_KEY", "")
    if _client is None and key:
        _client = cohere.Client(api_key=key)
    return _client


def _embed_with_retry(client, texts: list[str], input_type: str) -> list:
    """Embed with exponential backoff on rate limit errors."""
    for attempt in range(MAX_RETRIES):
        try:
            response = client.embed(
                texts=texts,
                model=EMBEDDING_MODEL,
                input_type=input_type,
                truncate="END",
            )
            return response.embeddings
        except Exception as e:
            if "429" in str(e) or "rate" in str(e).lower():
                wait = 2 ** attempt * 5
                time.sleep(wait)
            else:
                raise e
    raise RuntimeError("Cohere Embedding API rate limited. Vui lòng đợi 1 phút rồi thử lại.")


def embed_documents(texts: list[str], api_key: str = "") -> dict:
    """Embed document passages via Cohere API in batches."""
    client = _get_client(api_key)
    all_embeddings = []

    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i:i + EMBED_BATCH_SIZE]
        embeddings = _embed_with_retry(client, batch, "search_document")
        all_embeddings.extend(embeddings)

    return {"dense": all_embeddings}


def embed_batch(texts: list[str], api_key: str = "") -> list[list[float]]:
    """Embed a single batch."""
    result = embed_documents(texts, api_key)
    return result["dense"]


def embed_query(query: str, api_key: str = "") -> dict:
    """Embed a search query via Cohere API."""
    client = _get_client(api_key)
    embeddings = _embed_with_retry(client, [query], "search_query")
    return {"dense": embeddings[0]}
