"""Embedding via Cohere API - round-robin across keys, low memory."""

import time

import cohere

EMBED_BATCH_SIZE = 96
MAX_RETRIES = 3
EMBEDDING_MODEL = "embed-multilingual-v3.0"


def _load_all_cohere_keys() -> list[str]:
    keys = []
    try:
        from core.vectorstore import get_api_key
        for i in range(1, 11):
            key = get_api_key(f"cohere_{i}")
            if key:
                keys.append(key)
    except Exception:
        pass
    if not keys:
        try:
            import streamlit as st
            if hasattr(st, "secrets"):
                for k, v in st.secrets.items():
                    if k.startswith("COHERE_API_KEY") and v:
                        keys.append(v)
        except Exception:
            pass
    if not keys:
        import os
        for suffix in ["", "_2", "_3", "_4", "_5"]:
            key = os.environ.get(f"COHERE_API_KEY{suffix}", "")
            if key:
                keys.append(key)
    return keys


_clients = []
_current_key_idx = 0


def _get_clients(extra_key: str = "") -> list:
    global _clients
    if not _clients:
        keys = _load_all_cohere_keys()
        if extra_key and extra_key not in keys:
            keys.append(extra_key)
        _clients = [cohere.ClientV2(api_key=k) for k in keys]
    return _clients


def reset_clients():
    global _clients, _current_key_idx
    _clients = []
    _current_key_idx = 0


def _embed_one_batch(texts: list[str], input_type: str, extra_key: str = "") -> list:
    """Embed one batch with round-robin key selection + fallback."""
    global _current_key_idx
    clients = _get_clients(extra_key)

    if not clients:
        raise RuntimeError("Chưa có Cohere API key. Vào Settings để thêm.")

    num_keys = len(clients)
    for attempt in range(num_keys * MAX_RETRIES):
        client = clients[_current_key_idx % num_keys]
        try:
            response = client.embed(
                texts=texts,
                model=EMBEDDING_MODEL,
                input_type=input_type,
                embedding_types=["float"],
                truncate="END",
            )
            # Round-robin: next batch uses next key
            _current_key_idx += 1
            return [e for e in response.embeddings.float_]
        except Exception as e:
            err = str(e).lower()
            if "429" in str(e) or "rate" in err or "quota" in err or "limit" in err:
                _current_key_idx += 1
                if attempt % num_keys == num_keys - 1:
                    time.sleep(3)
            else:
                raise e

    raise RuntimeError("Tất cả Cohere keys đều hết quota.")


def embed_documents(texts: list[str], api_key: str = "") -> dict:
    all_embeddings = []
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i:i + EMBED_BATCH_SIZE]
        result = _embed_one_batch(batch, "search_document", api_key)
        all_embeddings.extend(result)
    return {"dense": all_embeddings}


def embed_batch(texts: list[str], api_key: str = "") -> list[list[float]]:
    result = embed_documents(texts, api_key)
    return result["dense"]


def embed_query(query: str, api_key: str = "") -> dict:
    result = _embed_one_batch([query], "search_query", api_key)
    return {"dense": result[0]}
