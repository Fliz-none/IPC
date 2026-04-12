from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import (
    CHILD_CHUNK_SIZE,
    CHILD_CHUNK_OVERLAP,
    PARENT_CHUNK_SIZE,
    PARENT_CHUNK_OVERLAP,
    CHUNK_SEPARATORS,
)


def chunk_pages(
    pages: list[tuple[int, str]], source_file: str
) -> list[dict]:
    """Split pages using parent-child chunking strategy.

    - Parent chunks (1024 chars): used as LLM context for complete information
    - Child chunks (256 chars): used for precise retrieval (indexed in vector DB)

    Each child chunk dict has keys: text, parent_text, metadata.
    """
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=PARENT_CHUNK_SIZE,
        chunk_overlap=PARENT_CHUNK_OVERLAP,
        separators=CHUNK_SEPARATORS,
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHILD_CHUNK_SIZE,
        chunk_overlap=CHILD_CHUNK_OVERLAP,
        separators=CHUNK_SEPARATORS,
    )

    chunks = []
    chunk_index = 0

    for page_number, text in pages:
        parent_splits = parent_splitter.split_text(text)

        for parent_text in parent_splits:
            child_splits = child_splitter.split_text(parent_text)

            for child_text in child_splits:
                chunks.append({
                    "text": child_text,
                    "parent_text": parent_text,
                    "metadata": {
                        "source_file": source_file,
                        "page_number": page_number,
                        "chunk_index": chunk_index,
                    },
                })
                chunk_index += 1

    return chunks
