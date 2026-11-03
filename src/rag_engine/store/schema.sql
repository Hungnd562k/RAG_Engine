-- Schema v1 (Docs/03-data-model.md). {dim} được thay bằng RAG_EMBED_DIM.
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id            BIGSERIAL PRIMARY KEY,
    source        TEXT        NOT NULL UNIQUE,
    content_hash  TEXT        NOT NULL,
    title         TEXT,
    metadata      JSONB       NOT NULL DEFAULT '{{}}',
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chunks (
    id            BIGSERIAL PRIMARY KEY,
    document_id   BIGINT      NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index   INT         NOT NULL,
    content       TEXT        NOT NULL,
    heading_path  TEXT[]      NOT NULL DEFAULT '{{}}',
    char_start    INT         NOT NULL,
    char_end      INT         NOT NULL,
    token_count   INT,
    embedding     vector({dim}) NOT NULL,
    metadata      JSONB       NOT NULL DEFAULT '{{}}',
    UNIQUE (document_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
    ON chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS chunks_document_id_idx ON chunks(document_id);

-- M6a hybrid search: full-text với config 'simple' (Postgres không có từ điển tiếng Việt;
-- 'simple' giữ nguyên dấu và mã định danh như `recon.daily`, `RAG_MIN_SCORE`).
-- ADD COLUMN IF NOT EXISTS → DB cũ tự nâng cấp, không cần re-embed.
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS content_tsv tsvector
    GENERATED ALWAYS AS (to_tsvector('simple', content)) STORED;

CREATE INDEX IF NOT EXISTS chunks_content_tsv_gin ON chunks USING gin (content_tsv);

CREATE TABLE IF NOT EXISTS engine_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
