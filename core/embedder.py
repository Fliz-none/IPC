from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL

_model = None

EMBED_BATCH_SIZE = 64


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def embed_documents(texts: list[str]) -> dict:
    """Embed document passages in batches to handle large files."""
    model = _get_model()
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        batch_size=EMBED_BATCH_SIZE,
        show_progress_bar=False,
    )
    return {"dense": embeddings.tolist()}


def embed_batch(texts: list[str]) -> list[list[float]]:
    """Embed a single batch of texts. Returns list of vectors."""
    model = _get_model()
    embeddings = model.encode(texts, normalize_embeddings=True)
    return embeddings.tolist()


def embed_query(query: str) -> dict:
    """Embed a search query using BGE-M3."""
    model = _get_model()
    embedding = model.encode(query, normalize_embeddings=True)
    return {"dense": embedding.tolist()}
