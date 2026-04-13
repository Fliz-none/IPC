from core.pdf_extract import extract_text_from_pdf
from core.chunker import chunk_pages
from core.embedder import embed_batch
from core.vectorstore import (
    create_document, get_or_create_document, insert_chunks_batch,
    list_documents, delete_document,
)
from core.retriever import retrieve
from core.generator import generate_answer_stream

INGEST_BATCH_SIZE = 64


def ingest_pdf(pdf_path: str, filename: str, progress_callback=None, cohere_key: str = "") -> int:
    """Ingest with resume support. If file was partially ingested, continues from where it stopped."""
    if progress_callback:
        progress_callback("extract", 0, 1)

    pages = extract_text_from_pdf(pdf_path)
    if not pages:
        raise ValueError("Không thể trích xuất text từ PDF này.")

    if progress_callback:
        progress_callback("extract", 1, 1)
        progress_callback("chunk", 0, 1)

    chunks = chunk_pages(pages, filename)
    total_chunks = len(chunks)

    if progress_callback:
        progress_callback("chunk", 1, 1)

    # Check if document exists and has partial data
    doc_id, existing_count = get_or_create_document(filename, total_chunks)

    if existing_count >= total_chunks:
        # Already fully ingested
        if progress_callback:
            progress_callback("done", total_chunks, total_chunks)
        return total_chunks

    # Skip already ingested chunks
    start_from = existing_count
    if progress_callback and start_from > 0:
        progress_callback("embed", start_from, total_chunks)

    for start in range(start_from, total_chunks, INGEST_BATCH_SIZE):
        end = min(start + INGEST_BATCH_SIZE, total_chunks)
        batch = chunks[start:end]

        batch_texts = [c["text"] for c in batch]
        batch_parents = [c["parent_text"] for c in batch]
        batch_metas = [c["metadata"] for c in batch]

        batch_embeddings = embed_batch(batch_texts, api_key=cohere_key)
        insert_chunks_batch(doc_id, batch_texts, batch_embeddings, batch_metas, batch_parents)

        if progress_callback:
            progress_callback("embed", end, total_chunks)

    if progress_callback:
        progress_callback("done", total_chunks, total_chunks)

    return total_chunks


def ask_stream(
    query: str,
    keys: dict = None,
    preferred_provider: str = "gemini",
    model: str = "",
):
    """Streaming Q&A with auto-fallback."""
    keys = keys or {}
    cohere_key = keys.get("cohere", "")

    chunks = retrieve(query, cohere_key=cohere_key)
    if not chunks:
        def empty():
            yield "Không tìm thấy thông tin liên quan trong tài liệu."
        return empty(), []
    stream = generate_answer_stream(
        query, chunks, keys=keys, preferred_provider=preferred_provider, model=model
    )
    return stream, chunks


def get_documents() -> list[dict]:
    return list_documents()


def remove_document(file_hash: str):
    delete_document(file_hash)
