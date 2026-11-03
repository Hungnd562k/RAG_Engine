from typing import Protocol

from rag_engine.models import Chunk, Document, RetrievedChunk


class VectorStore(Protocol):
    def get_document_hash(self, source: str) -> str | None: ...

    def replace_document(
        self, doc: Document, chunks: list[Chunk], vectors: list[list[float]]
    ) -> None:
        """Thêm mới hoặc thay toàn bộ chunks của document trong một transaction."""
        ...

    def delete_document(self, source: str) -> None: ...

    def list_sources(self, prefix: str = "") -> list[str]: ...

    def search(
        self, vector: list[float], top_k: int, source_prefix: str | None = None
    ) -> list[RetrievedChunk]:
        """Top-k chunk gần nhất theo cosine, score = cosine similarity, giảm dần."""
        ...

    def hybrid_search(
        self, query: str, vector: list[float], candidates: int, source_prefix: str | None = None
    ) -> list[RetrievedChunk]:
        """Hợp của top-`candidates` theo cosine và top-`candidates` theo full-text.

        Mỗi chunk có score = cosine similarity và lexical = Σ idf(từ khớp) / Σ idf(mọi từ của
        câu hỏi), idf = ln(1 + (N − df + 0.5) / (df + 0.5)) như BM25. Không theo thứ tự nào.
        """
        ...

    def stats(self) -> dict: ...

    def reset(self) -> None: ...
