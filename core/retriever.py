"""Retrieval with Cohere Rerank API. Zero local RAM."""

import cohere

from config import TOP_K, RERANK_TOP_N, COHERE_API_KEY
from core.embedder import embed_query
from core.vectorstore import hybrid_search

_cohere_client = None


def _get_cohere(api_key: str = ""):
    global _cohere_client
    key = api_key or COHERE_API_KEY
    if _cohere_client is None and key:
        _cohere_client = cohere.Client(api_key=key)
    return _cohere_client


def retrieve(
    query: str,
    top_k: int = TOP_K,
    rerank_top_n: int = RERANK_TOP_N,
    cohere_key: str = "",
    gemini_key: str = "",
) -> list[dict]:
    """Retrieve: Gemini embed -> PostgreSQL hybrid search -> Cohere rerank.

    Falls back to no reranking if Cohere key is missing or quota exceeded.
    """
    # Step 1: Embed query via Gemini API
    query_output = embed_query(query, api_key=gemini_key)

    # Step 2: PostgreSQL hybrid search
    candidates = hybrid_search(
        dense_embedding=query_output["dense"],
        query_text=query,
        top_k=top_k,
    )

    if not candidates:
        return []

    # Step 3: Cohere Rerank (fallback to RRF score if unavailable)
    cohere_client = _get_cohere(cohere_key)
    if cohere_client:
        try:
            docs = [c["document"] for c in candidates]
            response = cohere_client.rerank(
                query=query,
                documents=docs,
                model="rerank-multilingual-v3.0",
                top_n=rerank_top_n,
            )
            reranked = []
            for result in response.results:
                candidate = candidates[result.index]
                candidate["rerank_score"] = result.relevance_score
                reranked.append(candidate)
            return reranked
        except Exception:
            # Quota exceeded or error - fallback to RRF order
            pass

    # Fallback: return top results by RRF score
    return candidates[:rerank_top_n]
