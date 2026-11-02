import time

from rag_engine.embedding.base import Embedder
from rag_engine.models import SearchResult
from rag_engine.store.base import VectorStore


class Retriever:
    """Embed query → top-k cosine → lọc theo min_score."""

    def __init__(self, embedder: Embedder, store: VectorStore, min_score: float = 0.0) -> None:
        self.embedder = embedder
        self.store = store
        self.min_score = min_score

    def retrieve(
        self,
        query: str,
        top_k: int,
        source_prefix: str | None = None,
        min_score: float | None = None,
    ) -> SearchResult:
        if not query.strip():
            raise ValueError("Query rỗng")
        threshold = self.min_score if min_score is None else min_score
        t0 = time.perf_counter()
        vector = self.embedder.embed([query])[0]
        t1 = time.perf_counter()
        hits = self.store.search(vector, top_k=top_k, source_prefix=source_prefix)
        t2 = time.perf_counter()
        results = [h for h in hits if h.score >= threshold]
        return SearchResult(
            results=results,
            timings_ms={
                "embed": (t1 - t0) * 1000,
                "search": (t2 - t1) * 1000,
                "total": (t2 - t0) * 1000,
            },
        )
