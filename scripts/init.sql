-- Enable extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS unaccent;

-- API Keys storage
CREATE TABLE IF NOT EXISTS api_keys (
    id SERIAL PRIMARY KEY,
    provider VARCHAR(20) NOT NULL UNIQUE,
    api_key TEXT NOT NULL,
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Documents table
CREATE TABLE IF NOT EXISTS documents (
    id SERIAL PRIMARY KEY,
    filename TEXT NOT NULL,
    file_hash VARCHAR(16) NOT NULL UNIQUE,
    chunk_count INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Chunks table
CREATE TABLE IF NOT EXISTS chunks (
    id SERIAL PRIMARY KEY,
    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    page_number INTEGER NOT NULL,
    child_text TEXT NOT NULL,
    parent_text TEXT NOT NULL,
    embedding vector(1024),
    ts_content tsvector,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Chat sessions
CREATE TABLE IF NOT EXISTS chat_sessions (
    id SERIAL PRIMARY KEY,
    title TEXT NOT NULL DEFAULT 'Cuộc trò chuyện mới',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Chat messages
CREATE TABLE IF NOT EXISTS chat_messages (
    id SERIAL PRIMARY KEY,
    session_id INTEGER NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
    role VARCHAR(20) NOT NULL,
    content TEXT NOT NULL,
    sources JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_chunks_embedding
    ON chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 200);

CREATE INDEX IF NOT EXISTS idx_chunks_ts_content
    ON chunks USING gin (ts_content);

CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks (document_id);
CREATE INDEX IF NOT EXISTS idx_chat_messages_session_id ON chat_messages (session_id);

-- Auto-generate tsvector
CREATE OR REPLACE FUNCTION update_ts_content() RETURNS trigger AS $$
BEGIN
    NEW.ts_content := to_tsvector('simple', unaccent(NEW.child_text));
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_update_ts_content ON chunks;
CREATE TRIGGER trg_update_ts_content
    BEFORE INSERT OR UPDATE OF child_text ON chunks
    FOR EACH ROW EXECUTE FUNCTION update_ts_content();

-- Auto-update session timestamp
CREATE OR REPLACE FUNCTION update_session_timestamp() RETURNS trigger AS $$
BEGIN
    UPDATE chat_sessions SET updated_at = NOW() WHERE id = NEW.session_id;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_update_session_timestamp ON chat_messages;
CREATE TRIGGER trg_update_session_timestamp
    AFTER INSERT ON chat_messages
    FOR EACH ROW EXECUTE FUNCTION update_session_timestamp();
