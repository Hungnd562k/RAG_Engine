import logging
from importlib import resources

import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

from rag_engine.models import Chunk, Document, RetrievedChunk

log = logging.getLogger(__name__)

SCHEMA_VERSION = "1"


class EmbeddingConfigMismatch(RuntimeError):
    """Dữ liệu trong DB được tạo bằng model/dim khác cấu hình hiện tại."""


class PgVectorStore:
    """VectorStore trên PostgreSQL + pgvector (ADR-0005, Docs/03-data-model.md).

    Dùng connection pool: an toàn khi nhiều thread (FastAPI) gọi song song; mỗi thao tác
    mượn một connection riêng nên transaction của ingest không lẫn với search.
    """

    def __init__(
        self,
        database_url: str,
        embed_model: str,
        embed_dim: int,
        ef_search: int = 40,
        verify_meta: bool = True,
        pool_min_size: int = 1,
        pool_max_size: int = 4,
    ) -> None:
        self.database_url = database_url
        self.embed_model = embed_model
        self.embed_dim = embed_dim
        self.ef_search = ef_search
        # Schema phải có trước khi register_vector (cần type `vector`) → tạo bằng connection riêng.
        with psycopg.connect(database_url, autocommit=True, connect_timeout=5) as conn:
            self._ensure_schema(conn, verify_meta)
        self.pool = ConnectionPool(
            database_url,
            min_size=pool_min_size,
            max_size=pool_max_size,
            kwargs={"autocommit": True, "connect_timeout": 5},
            configure=self._configure,
            open=True,
        )
        self.pool.wait(timeout=10)

    def _configure(self, conn: psycopg.Connection) -> None:
        register_vector(conn)
        conn.execute("SELECT set_config('hnsw.ef_search', %s, false)", (str(self.ef_search),))

    # ---------- schema & meta ----------
    def _ensure_schema(self, conn: psycopg.Connection, verify_meta: bool = True) -> None:
        sql = resources.files("rag_engine.store").joinpath("schema.sql").read_text()
        with conn.transaction():
            conn.execute(sql.format(dim=int(self.embed_dim)))
        if verify_meta:
            self._check_meta(conn)

    @staticmethod
    def _meta(conn: psycopg.Connection) -> dict[str, str]:
        return dict(conn.execute("SELECT key, value FROM engine_meta").fetchall())

    def _check_meta(self, conn: psycopg.Connection) -> None:
        meta = self._meta(conn)
        wanted = {
            "embed_model": self.embed_model,
            "embed_dim": str(self.embed_dim),
            "schema_version": SCHEMA_VERSION,
        }
        if not meta:
            with conn.transaction():
                for k, v in wanted.items():
                    conn.execute("INSERT INTO engine_meta(key, value) VALUES (%s, %s)", (k, v))
            return
        diff = {k: (meta.get(k), v) for k, v in wanted.items() if meta.get(k) != v}
        if diff:
            details = ", ".join(f"{k}: DB={a!r} config={b!r}" for k, (a, b) in diff.items())
            raise EmbeddingConfigMismatch(
                f"Cấu hình lệch với dữ liệu đã lưu ({details}). Chạy `rag reset --yes` rồi ingest lại."
            )

    def check_meta(self) -> None:
        with self.pool.connection() as conn:
            self._check_meta(conn)

    def extension_version(self) -> str | None:
        with self.pool.connection() as conn:
            row = conn.execute(
                "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
            ).fetchone()
        return row[0] if row else None

    # ---------- documents ----------
    def get_document_hash(self, source: str) -> str | None:
        with self.pool.connection() as conn:
            row = conn.execute(
                "SELECT content_hash FROM documents WHERE source = %s", (source,)
            ).fetchone()
        return row[0] if row else None

    def replace_document(
        self, doc: Document, chunks: list[Chunk], vectors: list[list[float]]
    ) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("Số chunks và vectors không khớp")
        with self.pool.connection() as conn, conn.transaction(), conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO documents (source, content_hash, title, metadata)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (source) DO UPDATE
                   SET content_hash = EXCLUDED.content_hash,
                       title = EXCLUDED.title,
                       metadata = EXCLUDED.metadata,
                       updated_at = now()
                RETURNING id
                """,
                (doc.source, doc.content_hash, doc.title, Jsonb(doc.metadata)),
            )
            doc_id = cur.fetchone()[0]
            cur.execute("DELETE FROM chunks WHERE document_id = %s", (doc_id,))
            cur.executemany(
                """
                INSERT INTO chunks (document_id, chunk_index, content, heading_path,
                                    char_start, char_end, token_count, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                [
                    (
                        doc_id,
                        c.index,
                        c.text,
                        list(c.heading_path),
                        c.char_start,
                        c.char_end,
                        c.token_count,
                        np.asarray(v, dtype=np.float32),
                    )
                    for c, v in zip(chunks, vectors, strict=True)
                ],
            )

    def delete_document(self, source: str) -> None:
        with self.pool.connection() as conn:
            conn.execute("DELETE FROM documents WHERE source = %s", (source,))

    def list_sources(self, prefix: str = "") -> list[str]:
        with self.pool.connection() as conn:
            rows = conn.execute(
                "SELECT source FROM documents WHERE starts_with(source, %s) ORDER BY source",
                (prefix,),
            ).fetchall()
        return [r[0] for r in rows]

    # ---------- search ----------
    def search(
        self, vector: list[float], top_k: int, source_prefix: str | None = None
    ) -> list[RetrievedChunk]:
        q = np.asarray(vector, dtype=np.float32)
        with self.pool.connection() as conn:
            rows = conn.execute(
                """
                SELECT c.id, d.source, c.content, c.heading_path,
                       1 - (c.embedding <=> %(q)s) AS score
                FROM chunks c
                JOIN documents d ON d.id = c.document_id
                WHERE %(prefix)s::text IS NULL OR starts_with(d.source, %(prefix)s::text)
                ORDER BY c.embedding <=> %(q)s
                LIMIT %(k)s
                """,
                {"q": q, "prefix": source_prefix, "k": top_k},
            ).fetchall()
        return [
            RetrievedChunk(
                chunk_id=r[0], source=r[1], text=r[2], heading_path=tuple(r[3]), score=float(r[4])
            )
            for r in rows
        ]

    def hybrid_search(
        self, query: str, vector: list[float], candidates: int, source_prefix: str | None = None
    ) -> list[RetrievedChunk]:
        q = np.asarray(vector, dtype=np.float32)
        # Từ của câu hỏi lấy bằng chính config 'simple' như cột content_tsv. df đếm qua GIN index.
        # plainto_tsquery(t) dựng lại tsquery an toàn từ lexeme (không phải tự escape).
        with self.pool.connection() as conn:
            rows = conn.execute(
                """
                WITH terms AS (
                    SELECT DISTINCT unnest(tsvector_to_array(to_tsvector('simple', %(t)s))) AS t
                ),
                n AS (SELECT count(*)::float8 AS n FROM chunks),
                w AS (
                    SELECT tq, ln(1 + (n.n - df + 0.5) / (df + 0.5)) AS idf
                    FROM (
                        SELECT plainto_tsquery('simple', t) AS tq,
                               (SELECT count(*) FROM chunks
                                 WHERE content_tsv @@ plainto_tsquery('simple', t)) AS df
                        FROM terms
                    ) x CROSS JOIN n
                ),
                total AS (SELECT sum(idf) AS s FROM w),
                filtered AS (
                    SELECT c.id, c.embedding, c.content_tsv
                    FROM chunks c JOIN documents d ON d.id = c.document_id
                    WHERE %(prefix)s::text IS NULL OR starts_with(d.source, %(prefix)s::text)
                ),
                dense AS (
                    SELECT id FROM filtered ORDER BY embedding <=> %(q)s LIMIT %(k)s
                ),
                lex AS (
                    SELECT f.id FROM filtered f JOIN w ON f.content_tsv @@ w.tq
                    GROUP BY f.id ORDER BY sum(w.idf) DESC, f.id LIMIT %(k)s
                ),
                cand AS (SELECT id FROM dense UNION SELECT id FROM lex)
                SELECT c.id, d.source, c.content, c.heading_path,
                       1 - (c.embedding <=> %(q)s) AS score,
                       coalesce((SELECT sum(w.idf) FROM w WHERE c.content_tsv @@ w.tq), 0)
                         / nullif((SELECT s FROM total), 0) AS lexical
                FROM cand
                JOIN chunks c ON c.id = cand.id
                JOIN documents d ON d.id = c.document_id
                """,
                {"t": query, "q": q, "prefix": source_prefix, "k": candidates},
            ).fetchall()
        return [
            RetrievedChunk(
                chunk_id=r[0], source=r[1], text=r[2], heading_path=tuple(r[3]),
                score=float(r[4]), lexical=float(r[5] or 0.0),
            )
            for r in rows
        ]

    # ---------- admin ----------
    def stats(self) -> dict:
        with self.pool.connection() as conn:
            docs, chunks = conn.execute(
                "SELECT (SELECT count(*) FROM documents), (SELECT count(*) FROM chunks)"
            ).fetchone()
            meta = self._meta(conn)
            version = conn.execute(
                "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
            ).fetchone()
        return {
            "documents": docs,
            "chunks": chunks,
            "embed_model": meta.get("embed_model"),
            "embed_dim": int(meta["embed_dim"]) if "embed_dim" in meta else None,
            "pgvector": version[0] if version else None,
        }

    def reset(self) -> None:
        """Xóa toàn bộ dữ liệu và schema, tạo lại theo cấu hình hiện tại."""
        with self.pool.connection() as conn:
            with conn.transaction():
                conn.execute("DROP TABLE IF EXISTS chunks, documents, engine_meta CASCADE")
            self._ensure_schema(conn)

    def close(self) -> None:
        self.pool.close()
