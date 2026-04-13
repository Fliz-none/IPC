"""Embedding via Gemini Embedding API. Zero local RAM."""

from google import genai

from config import GEMINI_API_KEY, EMBEDDING_MODEL, DENSE_VECTOR_SIZE

_client = None

EMBED_BATCH_SIZE = 100


def _get_client(api_key: str = "") -> genai.Client:
    global _client
    key = api_key or GEMINI_API_KEY
    if _client is None and key:
        _client = genai.Client(api_key=key)
    return _client


def embed_documents(texts: list[str], api_key: str = "") -> dict:
    """Embed document passages via Gemini API in batches."""
    client = _get_client(api_key)
    all_embeddings = []

    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i:i + EMBED_BATCH_SIZE]
        response = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=batch,
            config=genai.types.EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT",
                output_dimensionality=DENSE_VECTOR_SIZE,
            ),
        )
        for emb in response.embeddings:
            all_embeddings.append(emb.values)

    return {"dense": all_embeddings}


def embed_batch(texts: list[str], api_key: str = "") -> list[list[float]]:
    """Embed a single batch. Returns list of vectors."""
    result = embed_documents(texts, api_key)
    return result["dense"]


def embed_query(query: str, api_key: str = "") -> dict:
    """Embed a search query via Gemini API."""
    client = _get_client(api_key)
    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=query,
        config=genai.types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY",
            output_dimensionality=DENSE_VECTOR_SIZE,
        ),
    )
    return {"dense": response.embeddings[0].values}
