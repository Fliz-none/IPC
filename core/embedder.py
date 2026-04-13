"""Embedding via Cohere API - parallel across multiple keys for max speed."""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import cohere

EMBED_BATCH_SIZE = 96
MAX_RETRIES = 3
EMBEDDING_MODEL = "embed-multilingual-v3.0"


def _load_all_cohere_keys() -> list[str]:
    """Load all Cohere keys from DB, then Streamlit secrets, then env."""
    keys = []
    try:
        from core.vectorstore import get_api_key
        for i in range(1, 11):  # Support up to 10 keys
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
        _clients = [cohere.Client(api_key=k) for k in keys]
    return _clients


def reset_clients():
    global _clients, _current_key_idx
    _clients = []
    _current_key_idx = 0


def _embed_single(client, texts: list[str], input_type: str) -> list:
    """Embed using a single client with retry."""
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
            err = str(e).lower()
            if "429" in str(e) or "rate" in err or "quota" in err or "limit" in err:
                time.sleep(2 ** attempt * 3)
            else:
                raise e
    return None  # All retries failed


def _embed_with_rotation(texts: list[str], input_type: str, extra_key: str = "") -> list:
    """Single batch embed with key rotation fallback."""
    global _current_key_idx
    clients = _get_clients(extra_key)

    if not clients:
        raise RuntimeError("Chưa có Cohere API key. Vào Settings để thêm.")

    for _ in range(len(clients)):
        client = clients[_current_key_idx % len(clients)]
        result = _embed_single(client, texts, input_type)
        if result is not None:
            return result
        _current_key_idx += 1

    raise RuntimeError("Tất cả Cohere keys đều hết quota. Thêm key mới trong Settings.")


def embed_documents(texts: list[str], api_key: str = "") -> dict:
    """Embed documents - parallel across keys if multiple available."""
    clients = _get_clients(api_key)

    if not clients:
        raise RuntimeError("Chưa có Cohere API key. Vào Settings để thêm.")

    # Split texts into batches
    batches = []
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batches.append(texts[i:i + EMBED_BATCH_SIZE])

    num_keys = len(clients)

    if num_keys == 1:
        # Single key - sequential
        all_embeddings = []
        for batch in batches:
            result = _embed_with_rotation(batch, "search_document", api_key)
            all_embeddings.extend(result)
        return {"dense": all_embeddings}

    # Multiple keys - parallel: distribute batches across keys
    all_embeddings = [None] * len(batches)

    def process_batch(batch_idx: int, batch: list[str], client_idx: int):
        client = clients[client_idx % num_keys]
        result = _embed_single(client, batch, "search_document")
        if result is None:
            # Retry with next key
            for fallback in range(num_keys):
                alt_client = clients[(client_idx + fallback + 1) % num_keys]
                result = _embed_single(alt_client, batch, "search_document")
                if result is not None:
                    break
        return batch_idx, result

    with ThreadPoolExecutor(max_workers=num_keys) as executor:
        futures = []
        for i, batch in enumerate(batches):
            # Round-robin assign batches to keys
            futures.append(executor.submit(process_batch, i, batch, i))

        for future in as_completed(futures):
            batch_idx, result = future.result()
            if result is None:
                raise RuntimeError(f"Batch {batch_idx} thất bại. Tất cả keys hết quota.")
            all_embeddings[batch_idx] = result

    # Flatten
    flat = []
    for batch_result in all_embeddings:
        flat.extend(batch_result)
    return {"dense": flat}


def embed_batch(texts: list[str], api_key: str = "") -> list[list[float]]:
    result = embed_documents(texts, api_key)
    return result["dense"]


def embed_query(query: str, api_key: str = "") -> dict:
    embeddings = _embed_with_rotation([query], "search_query", api_key)
    return {"dense": embeddings[0]}
