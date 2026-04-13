"""Embedding via Cohere API with multi-key rotation on rate limits."""

import time
import os

import cohere

EMBED_BATCH_SIZE = 96
MAX_RETRIES = 3
EMBEDDING_MODEL = "embed-multilingual-v3.0"


def _load_all_cohere_keys() -> list[str]:
    """Load all Cohere keys: COHERE_API_KEY, COHERE_API_KEY_2, COHERE_API_KEY_3, ..."""
    keys = []
    # From Streamlit secrets
    try:
        import streamlit as st
        if hasattr(st, "secrets"):
            for k, v in st.secrets.items():
                if k.startswith("COHERE_API_KEY") and v:
                    keys.append(v)
    except Exception:
        pass
    # From env vars (fallback / Docker)
    if not keys:
        for suffix in ["", "_2", "_3", "_4", "_5"]:
            key = os.environ.get(f"COHERE_API_KEY{suffix}", "")
            if key:
                keys.append(key)
    # From DB (saved via Settings page)
    if not keys:
        try:
            from core.vectorstore import get_api_key
            key = get_api_key("cohere")
            if key:
                keys.append(key)
        except Exception:
            pass
    return keys


_clients = []
_current_key_idx = 0


def _get_clients(extra_key: str = "") -> list:
    global _clients
    if not _clients:
        keys = _load_all_cohere_keys()
        if extra_key and extra_key not in keys:
            keys.append(extra_key)
        _clients = [cohere.Client(api_key=k) for k in keys]
    return _clients


def _embed_with_rotation(texts: list[str], input_type: str, extra_key: str = "") -> list:
    """Embed with auto-rotation across multiple Cohere keys on rate limit."""
    global _current_key_idx
    clients = _get_clients(extra_key)

    if not clients:
        raise RuntimeError("Chưa có Cohere API key. Vào Settings để thêm.")

    tried = 0
    while tried < len(clients) * MAX_RETRIES:
        client = clients[_current_key_idx % len(clients)]
        try:
            response = client.embed(
                texts=texts,
                model=EMBEDDING_MODEL,
                input_type=input_type,
                truncate="END",
            )
            return response.embeddings
        except Exception as e:
            err = str(e).lower()
            if "429" in str(e) or "rate" in err or "quota" in err or "limit" in err:
                # Switch to next key
                _current_key_idx += 1
                if _current_key_idx % len(clients) == 0:
                    # All keys exhausted this round, wait before retry
                    time.sleep(5)
                tried += 1
            else:
                raise e

    raise RuntimeError("Tất cả Cohere keys đều hết quota. Thêm key mới trong Settings.")


def embed_documents(texts: list[str], api_key: str = "") -> dict:
    """Embed document passages with multi-key rotation."""
    all_embeddings = []
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i:i + EMBED_BATCH_SIZE]
        embeddings = _embed_with_rotation(batch, "search_document", api_key)
        all_embeddings.extend(embeddings)
    return {"dense": all_embeddings}


def embed_batch(texts: list[str], api_key: str = "") -> list[list[float]]:
    result = embed_documents(texts, api_key)
    return result["dense"]


def embed_query(query: str, api_key: str = "") -> dict:
    embeddings = _embed_with_rotation([query], "search_query", api_key)
    return {"dense": embeddings[0]}
