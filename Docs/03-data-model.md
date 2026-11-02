# 03 — Data Model

## 1. Schema (PostgreSQL 16 + pgvector)

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id            BIGSERIAL PRIMARY KEY,
    source        TEXT        NOT NULL UNIQUE,     -- đường dẫn tương đối, khóa tự nhiên
    content_hash  TEXT        NOT NULL,            -- sha256 nội dung file
    title         TEXT,
    metadata      JSONB       NOT NULL DEFAULT '{}',
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chunks (
    id            BIGSERIAL PRIMARY KEY,
    document_id   BIGINT      NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index   INT         NOT NULL,
    content       TEXT        NOT NULL,
    heading_path  TEXT[]      NOT NULL DEFAULT '{}',
    char_start    INT         NOT NULL,
    char_end      INT         NOT NULL,
    token_count   INT,
    embedding     vector(1024) NOT NULL,
    metadata      JSONB       NOT NULL DEFAULT '{}',
    UNIQUE (document_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
    ON chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS chunks_document_id_idx ON chunks(document_id);

-- Metadata hệ thống: khóa model/dim đã dùng để phát hiện lệch cấu hình
CREATE TABLE IF NOT EXISTS engine_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
-- vd: ('embed_model','bge-m3'), ('embed_dim','1024'), ('schema_version','1')
```

## 2. Truy vấn tìm kiếm

```sql
-- đặt một lần cho mỗi connection (RAG_HNSW_EF_SEARCH); tăng để recall cao hơn, đổi lại chậm hơn
SELECT set_config('hnsw.ef_search', '40', false);

SELECT c.id, d.source, c.content, c.heading_path,
       1 - (c.embedding <=> %(q)s) AS score
FROM chunks c
JOIN documents d ON d.id = c.document_id
WHERE %(prefix)s::text IS NULL OR starts_with(d.source, %(prefix)s::text)
ORDER BY c.embedding <=> %(q)s
LIMIT %(k)s;
```
- `<=>` là cosine distance; `score = 1 - distance` (cosine similarity).
- Lọc `min_score` ở tầng ứng dụng sau khi lấy top-k (giữ index được dùng).

## 3. Chiến lược ingest idempotent

| Trường hợp | Điều kiện | Hành động |
|------------|-----------|-----------|
| Mới | `source` chưa có | INSERT document + chunks |
| Không đổi | `content_hash` trùng | Skip (không gọi embedding) |
| Đã đổi | `content_hash` khác | Trong 1 transaction: DELETE chunks cũ → UPDATE document → INSERT chunks mới |
| Bị xóa khỏi thư mục | (tùy chọn `--prune`) | DELETE document (cascade chunks) |

- Embedding được tính **trước** khi mở transaction để transaction ngắn.
- Khi khởi động, so `engine_meta.embed_model/embed_dim` với config; lệch → báo lỗi rõ ràng, yêu cầu `rag reset` (tránh trộn vector từ hai model).

## 4. Migration
MVP dùng một file `schema.sql` idempotent (`IF NOT EXISTS`), chạy tự động khi `rag health`/`rag ingest`. Khi schema cần đổi (M6+), chuyển sang công cụ migration (vd. Alembic hoặc yoyo) và tăng `schema_version`.

## 5. Ước lượng dung lượng
`vector(1024)` float4 ≈ 4 KB/chunk + nội dung. 100k chunks ≈ 400 MB vector + index HNSW (~1.5–2×). Dư sức cho máy local.
