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

    def stats(self) -> dict: ...

    def reset(self) -> None: ...
