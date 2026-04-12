from core.pdf_extract import extract_text_from_pdf
from core.chunker import chunk_pages
from core.embedder import embed_batch
from core.vectorstore import (
    create_document, insert_chunks_batch,
    list_documents, delete_document,
)
from core.retriever import retrieve
from core.generator import generate_answer_stream

INGEST_BATCH_SIZE = 64


def ingest_pdf(pdf_path: str, filename: str, progress_callback=None) -> int:
    """Batch ingestion pipeline for large files.

    progress_callback(stage, current, total) is called to report progress.
    Stages: "extract", "chunk", "embed", "store", "done"
    """
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

    # Create document record
    doc_id = create_document(filename, total_chunks)

    # Process in batches: embed + insert
    for start in range(0, total_chunks, INGEST_BATCH_SIZE):
        end = min(start + INGEST_BATCH_SIZE, total_chunks)
        batch = chunks[start:end]

        batch_texts = [c["text"] for c in batch]
        batch_parents = [c["parent_text"] for c in batch]
        batch_metas = [c["metadata"] for c in batch]

        # Embed batch
        batch_embeddings = embed_batch(batch_texts)

        # Insert batch into DB
        insert_chunks_batch(doc_id, batch_texts, batch_embeddings, batch_metas, batch_parents)

        if progress_callback:
            progress_callback("embed", end, total_chunks)

    if progress_callback:
        progress_callback("done", total_chunks, total_chunks)

    return total_chunks


def ask_stream(query: str, model: str = "", api_key: str = ""):
    """Streaming Q&A pipeline."""
    chunks = retrieve(query)
    if not chunks:
        def empty():
            yield "Không tìm thấy thông tin liên quan trong tài liệu."
        return empty(), []
    stream = generate_answer_stream(query, chunks, model=model, api_key=api_key)
    return stream, chunks


def get_documents() -> list[dict]:
    return list_documents()


def remove_document(file_hash: str):
    delete_document(file_hash)
