"""Retrieval: Cohere Embed + PostgreSQL hybrid search. No reranking."""

from config import TOP_K, RERANK_TOP_N
from core.embedder import embed_query
from core.vectorstore import hybrid_search


def retrieve(
    query: str,
    top_k: int = TOP_K,
    rerank_top_n: int = RERANK_TOP_N,
    cohere_key: str = "",
    **kwargs,
) -> list[dict]:
    """Cohere embed -> PostgreSQL hybrid search (vector + full-text RRF)."""
    # Step 1: Embed query via Cohere
    query_output = embed_query(query, api_key=cohere_key)

    # Step 2: PostgreSQL hybrid search (RRF fusion of vector + tsvector)
    candidates = hybrid_search(
        dense_embedding=query_output["dense"],
        query_text=query,
        top_k=top_k,
    )

    return candidates[:rerank_top_n]
