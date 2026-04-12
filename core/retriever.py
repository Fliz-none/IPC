from sentence_transformers import CrossEncoder

from config import TOP_K, RERANK_TOP_N
from core.embedder import embed_query
from core.vectorstore import hybrid_search

_reranker = None
RERANK_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"


def _get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(RERANK_MODEL)
    return _reranker


def retrieve(
    query: str,
    top_k: int = TOP_K,
    rerank_top_n: int = RERANK_TOP_N,
) -> list[dict]:
    """Retrieve relevant chunks using PostgreSQL hybrid search + cross-encoder reranking.

    Pipeline:
    1. BGE-M3 encodes query -> dense vector
    2. PostgreSQL hybrid search (vector cosine + tsvector full-text, RRF fusion) -> top_k
    3. Cross-encoder reranking -> top rerank_top_n results
    """
    # Step 1: Encode query
    query_output = embed_query(query)

    # Step 2: PostgreSQL hybrid search (vector + full-text, fused with RRF in SQL)
    candidates = hybrid_search(
        dense_embedding=query_output["dense"],
        query_text=query,
        top_k=top_k,
    )

    if not candidates:
        return []

    # Step 3: Cross-encoder reranking
    reranker = _get_reranker()
    pairs = [(query, c["document"]) for c in candidates]
    rerank_scores = reranker.predict(pairs)

    for i, score in enumerate(rerank_scores):
        candidates[i]["rerank_score"] = float(score)

    reranked = sorted(
        candidates, key=lambda x: x["rerank_score"], reverse=True
    )
    return reranked[:rerank_top_n]
