import time
from typing import Literal

from rag_engine.embedding.base import Embedder
from rag_engine.models import RetrievedChunk, SearchResult
from rag_engine.store.base import VectorStore

SearchMode = Literal["dense", "hybrid"]


def fuse(candidates: list[RetrievedChunk], alpha: float) -> list[tuple[float, RetrievedChunk]]:
    """Điểm gộp = cosine + alpha × lexical, giảm dần (hòa → chunk_id tăng dần).

    Không dùng RRF: RRF luôn thưởng chunk có mặt ở cả hai danh sách, nên câu hỏi khác ngôn ngữ
    với tài liệu (full-text không thể khớp) bị chunk "trông giống" cùng ngôn ngữ đẩy khỏi top-k
    (đo ở M6a, Docs/06 §6). Cộng có trọng số nhỏ chỉ đảo thứ tự khi cosine sát nhau.
    """
    scored = [(h.score + alpha * (h.lexical or 0.0), h) for h in candidates]
    scored.sort(key=lambda x: (-x[0], x[1].chunk_id))
    return scored


class Retriever:
    """Embed query → top-k → lọc theo min_score.

    - dense: top-k cosine, bỏ chunk có cosine < min_score.
    - hybrid: ứng viên = top-`candidates` cosine ∪ top-`candidates` full-text; xếp theo
      cosine + alpha × độ phủ từ khóa (IDF). Câu hỏi bị chặn (trả rỗng) khi cosine cao nhất
      < min_score — giữ nguyên cách chặn câu ngoài phạm vi của dense. Nếu không bị chặn,
      giữ chunk có điểm gộp ≥ min_score.

    `score` của chunk trả về luôn là cosine; `top_score` là cosine cao nhất (điểm để chặn).
    """

    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        min_score: float = 0.0,
        mode: SearchMode = "dense",
        alpha: float = 0.15,
        candidates: int = 20,
    ) -> None:
        self.embedder = embedder
        self.store = store
        self.min_score = min_score
        self.mode = mode
        self.alpha = alpha
        self.candidates = candidates

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

        if self.mode == "hybrid":
            cands = self.store.hybrid_search(
                query, vector, candidates=max(self.candidates, top_k), source_prefix=source_prefix
            )
            top = max((h.score for h in cands), default=None)
            if top is None or top < threshold:
                results: list[RetrievedChunk] = []
            else:
                results = [h for s, h in fuse(cands, self.alpha) if s >= threshold][:top_k]
        else:
            hits = self.store.search(vector, top_k=top_k, source_prefix=source_prefix)
            top = hits[0].score if hits else None
            results = [h for h in hits if h.score >= threshold]
        t2 = time.perf_counter()

        return SearchResult(
            results=results,
            timings_ms={
                "embed": (t1 - t0) * 1000,
                "search": (t2 - t1) * 1000,
                "total": (t2 - t0) * 1000,
            },
            top_score=top,
        )
