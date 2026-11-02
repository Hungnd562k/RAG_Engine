import logging
from importlib import resources

import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from psycopg.types.json import Jsonb

from rag_engine.models import Chunk, Document, RetrievedChunk

log = logging.getLogger(__name__)

SCHEMA_VERSION = "1"


class EmbeddingConfigMismatch(RuntimeError):
    """Dữ liệu trong DB được tạo bằng model/dim khác cấu hình hiện tại."""


class PgVectorStore:
    """VectorStore trên PostgreSQL + pgvector (ADR-0005, Docs/03-data-model.md)."""

    def __init__(
        self,
        database_url: str,
        embed_model: str,
        embed_dim: int,
        ef_search: int = 40,
        verify_meta: bool = True,
    ) -> None:
        self.embed_model = embed_model
        self.embed_dim = embed_dim
        self.ef_search = ef_search
        self.conn = psycopg.connect(database_url, autocommit=True, connect_timeout=5)
        self._ensure_schema(verify_meta)
        register_vector(self.conn)
        self.conn.execute("SELECT set_config('hnsw.ef_search', %s, false)", (str(ef_search),))

    # ---------- schema & meta ----------
    def _ensure_schema(self, verify_meta: bool = True) -> None:
        sql = resources.files("rag_engine.store").joinpath("schema.sql").read_text()
        with self.conn.transaction():
            self.conn.execute(sql.format(dim=int(self.embed_dim)))
        if verify_meta:
            self.check_meta()

    def _meta(self) -> dict[str, str]:
        return dict(self.conn.execute("SELECT key, value FROM engine_meta").fetchall())

    def check_meta(self) -> None:
        meta = self._meta()
        wanted = {
            "embed_model": self.embed_model,
            "embed_dim": str(self.embed_dim),
            "schema_version": SCHEMA_VERSION,
        }
        if not meta:
            with self.conn.transaction():
                for k, v in wanted.items():
                    self.conn.execute("INSERT INTO engine_meta(key, value) VALUES (%s, %s)", (k, v))
            return
        diff = {k: (meta.get(k), v) for k, v in wanted.items() if meta.get(k) != v}
        if diff:
            details = ", ".join(f"{k}: DB={a!r} config={b!r}" for k, (a, b) in diff.items())
            raise EmbeddingConfigMismatch(
                f"Cấu hình lệch với dữ liệu đã lưu ({details}). Chạy `rag reset --yes` rồi ingest lại."
            )

    def extension_version(self) -> str | None:
        row = self.conn.execute(
            "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
        ).fetchone()
        return row[0] if row else None

    # ---------- documents ----------
    def get_document_hash(self, source: str) -> str | None:
        row = self.conn.execute(
            "SELECT content_hash FROM documents WHERE source = %s", (source,)
        ).fetchone()
        return row[0] if row else None

    def replace_document(
        self, doc: Document, chunks: list[Chunk], vectors: list[list[float]]
    ) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("Số chunks và vectors không khớp")
        with self.conn.transaction(), self.conn.cursor() as cur:
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
        self.conn.execute("DELETE FROM documents WHERE source = %s", (source,))

    def list_sources(self, prefix: str = "") -> list[str]:
        rows = self.conn.execute(
            "SELECT source FROM documents WHERE starts_with(source, %s) ORDER BY source",
            (prefix,),
        ).fetchall()
        return [r[0] for r in rows]

    # ---------- search ----------
    def search(
        self, vector: list[float], top_k: int, source_prefix: str | None = None
    ) -> list[RetrievedChunk]:
        q = np.asarray(vector, dtype=np.float32)
        rows = self.conn.execute(
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

    # ---------- admin ----------
    def stats(self) -> dict:
        docs, chunks = self.conn.execute(
            "SELECT (SELECT count(*) FROM documents), (SELECT count(*) FROM chunks)"
        ).fetchone()
        meta = self._meta()
        return {
            "documents": docs,
            "chunks": chunks,
            "embed_model": meta.get("embed_model"),
            "embed_dim": int(meta["embed_dim"]) if "embed_dim" in meta else None,
            "pgvector": self.extension_version(),
        }

    def reset(self) -> None:
        """Xóa toàn bộ dữ liệu và schema, tạo lại theo cấu hình hiện tại."""
        with self.conn.transaction():
            self.conn.execute("DROP TABLE IF EXISTS chunks, documents, engine_meta CASCADE")
        self._ensure_schema()

    def close(self) -> None:
        self.conn.close()
