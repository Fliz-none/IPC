import hashlib
import json

import psycopg2
from psycopg2.extras import execute_values
from pgvector.psycopg2 import register_vector

from config import DATABASE_URL

_conn = None


def _get_conn():
    global _conn
    if _conn is None or _conn.closed:
        _conn = psycopg2.connect(DATABASE_URL)
        _conn.autocommit = True
        register_vector(_conn)
    return _conn


def _file_hash(filename: str) -> str:
    return hashlib.md5(filename.encode()).hexdigest()[:16]


# --- Documents ---

def create_document(source_file: str, chunk_count: int) -> int:
    conn = _get_conn()
    fhash = _file_hash(source_file)
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO documents (filename, file_hash, chunk_count)
            VALUES (%s, %s, %s)
            ON CONFLICT (file_hash) DO UPDATE
                SET filename = EXCLUDED.filename,
                    chunk_count = EXCLUDED.chunk_count,
                    created_at = NOW()
            RETURNING id
            """,
            (source_file, fhash, chunk_count),
        )
        doc_id = cur.fetchone()[0]
        cur.execute("DELETE FROM chunks WHERE document_id = %s", (doc_id,))
    return doc_id


def insert_chunks_batch(
    doc_id: int,
    chunks: list[str],
    embeddings: list[list[float]],
    metadatas: list[dict],
    parent_texts: list[str],
):
    conn = _get_conn()
    rows = []
    for chunk, emb, meta, parent in zip(chunks, embeddings, metadatas, parent_texts):
        rows.append((
            doc_id,
            meta.get("chunk_index", 0),
            meta.get("page_number", 0),
            chunk,
            parent,
            emb,
        ))

    with conn.cursor() as cur:
        execute_values(
            cur,
            """
            INSERT INTO chunks
                (document_id, chunk_index, page_number, child_text, parent_text, embedding)
            VALUES %s
            """,
            rows,
            template="(%s, %s, %s, %s, %s, %s::vector)",
        )


def add_document(
    source_file: str,
    chunks: list[str],
    embeddings: list[list[float]],
    metadatas: list[dict],
    parent_texts: list[str],
) -> int:
    doc_id = create_document(source_file, len(chunks))
    insert_chunks_batch(doc_id, chunks, embeddings, metadatas, parent_texts)
    return len(chunks)


def hybrid_search(
    dense_embedding: list[float],
    query_text: str,
    top_k: int,
    source_filter: str | None = None,
) -> list[dict]:
    conn = _get_conn()

    filter_clause = ""
    if source_filter:
        filter_clause = "AND c.document_id IN (SELECT id FROM documents WHERE filename = %s)"

    sql = f"""
    WITH vector_search AS (
        SELECT c.id, c.child_text, c.parent_text, c.page_number, c.chunk_index,
               d.filename AS source_file,
               ROW_NUMBER() OVER (ORDER BY c.embedding <=> %s::vector) AS rank
        FROM chunks c
        JOIN documents d ON c.document_id = d.id
        WHERE 1=1 {filter_clause if source_filter else ''}
        LIMIT %s
    ),
    text_search AS (
        SELECT c.id, c.child_text, c.parent_text, c.page_number, c.chunk_index,
               d.filename AS source_file,
               ROW_NUMBER() OVER (
                   ORDER BY ts_rank(c.ts_content, plainto_tsquery('simple', unaccent(%s))) DESC
               ) AS rank
        FROM chunks c
        JOIN documents d ON c.document_id = d.id
        WHERE c.ts_content @@ plainto_tsquery('simple', unaccent(%s))
        {filter_clause if source_filter else ''}
        LIMIT %s
    ),
    fused AS (
        SELECT
            COALESCE(v.id, t.id) AS id,
            COALESCE(v.child_text, t.child_text) AS child_text,
            COALESCE(v.parent_text, t.parent_text) AS parent_text,
            COALESCE(v.page_number, t.page_number) AS page_number,
            COALESCE(v.chunk_index, t.chunk_index) AS chunk_index,
            COALESCE(v.source_file, t.source_file) AS source_file,
            COALESCE(1.0 / (60 + v.rank), 0.0) +
            COALESCE(1.0 / (60 + t.rank), 0.0) AS rrf_score
        FROM vector_search v
        FULL OUTER JOIN text_search t ON v.id = t.id
    )
    SELECT child_text, parent_text, page_number, chunk_index, source_file, rrf_score
    FROM fused
    ORDER BY rrf_score DESC
    LIMIT %s
    """

    if source_filter:
        params = [
            dense_embedding, source_filter, top_k,
            query_text, query_text, source_filter, top_k,
            top_k,
        ]
    else:
        params = [
            dense_embedding, top_k,
            query_text, query_text, top_k,
            top_k,
        ]

    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()

    return [
        {
            "document": r[0],
            "parent_text": r[1],
            "metadata": {
                "page_number": r[2],
                "chunk_index": r[3],
                "source_file": r[4],
            },
            "score": float(r[5]),
        }
        for r in rows
    ]


def list_documents() -> list[dict]:
    conn = _get_conn()
    with conn.cursor() as cur:
        cur.execute(
            "SELECT filename, file_hash, chunk_count, created_at FROM documents ORDER BY created_at DESC"
        )
        rows = cur.fetchall()
    return [
        {"name": r[0], "hash": r[1], "chunks": r[2], "created_at": str(r[3])}
        for r in rows
    ]


def delete_document(file_hash: str):
    conn = _get_conn()
    with conn.cursor() as cur:
        cur.execute("DELETE FROM documents WHERE file_hash = %s", (file_hash,))


# --- Chat history ---

def create_chat_session(title: str = "Cuộc trò chuyện mới") -> int:
    conn = _get_conn()
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO chat_sessions (title) VALUES (%s) RETURNING id",
            (title,),
        )
        return cur.fetchone()[0]


def list_chat_sessions() -> list[dict]:
    conn = _get_conn()
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, title, created_at, updated_at FROM chat_sessions ORDER BY updated_at DESC"
        )
        rows = cur.fetchall()
    return [
        {"id": r[0], "title": r[1], "created_at": str(r[2]), "updated_at": str(r[3])}
        for r in rows
    ]


def get_chat_messages(session_id: int) -> list[dict]:
    conn = _get_conn()
    with conn.cursor() as cur:
        cur.execute(
            "SELECT role, content, sources FROM chat_messages WHERE session_id = %s ORDER BY id",
            (session_id,),
        )
        rows = cur.fetchall()
    return [{"role": r[0], "content": r[1], "sources": r[2]} for r in rows]


def save_chat_message(session_id: int, role: str, content: str, sources=None):
    conn = _get_conn()
    sources_json = json.dumps(sources) if sources else None
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO chat_messages (session_id, role, content, sources) VALUES (%s, %s, %s, %s)",
            (session_id, role, content, sources_json),
        )


def update_session_title(session_id: int, title: str):
    conn = _get_conn()
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE chat_sessions SET title = %s WHERE id = %s",
            (title, session_id),
        )


def delete_chat_session(session_id: int):
    conn = _get_conn()
    with conn.cursor() as cur:
        cur.execute("DELETE FROM chat_sessions WHERE id = %s", (session_id,))


# --- API Keys ---

def save_api_key(provider: str, api_key: str):
    conn = _get_conn()
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO api_keys (provider, api_key, updated_at)
            VALUES (%s, %s, NOW())
            ON CONFLICT (provider) DO UPDATE
                SET api_key = EXCLUDED.api_key, updated_at = NOW()
            """,
            (provider, api_key),
        )


def get_api_key(provider: str) -> str:
    conn = _get_conn()
    with conn.cursor() as cur:
        cur.execute(
            "SELECT api_key FROM api_keys WHERE provider = %s",
            (provider,),
        )
        row = cur.fetchone()
    return row[0] if row else ""


def delete_api_key(provider: str):
    conn = _get_conn()
    with conn.cursor() as cur:
        cur.execute("DELETE FROM api_keys WHERE provider = %s", (provider,))
