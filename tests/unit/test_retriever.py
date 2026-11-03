import pytest

from rag_engine.models import Chunk, Document, RetrievedChunk
from rag_engine.retrieval.retriever import Retriever, fuse


def rc(cid: int, score: float = 0.9, lexical: float | None = None) -> RetrievedChunk:
    return RetrievedChunk(cid, f"docs/{cid}.md", f"chunk {cid}", (), score, lexical)


class StubEmbedder:
    def embed(self, texts):
        return [[1.0] for _ in texts]


class StubStore:
    """Trả về danh sách dựng sẵn để kiểm tra logic gộp/lọc của Retriever."""

    def __init__(self, dense=(), hybrid=()):
        self.dense, self.hybrid = list(dense), list(hybrid)
        self.calls: list[tuple[str, int]] = []

    def search(self, vector, top_k, source_prefix=None):
        self.calls.append(("dense", top_k))
        return self.dense[:top_k]

    def hybrid_search(self, query, vector, candidates, source_prefix=None):
        self.calls.append(("hybrid", candidates))
        return self.hybrid


# ---------- fuse ----------
def test_fuse_small_alpha_only_reorders_close_cosines():
    # 1 vs 2: cosine sát nhau → từ khóa đảo thứ tự; 3: cosine vượt xa → giữ hạng 1
    cands = [rc(1, 0.60, 0.1), rc(2, 0.58, 0.9), rc(3, 0.75, 0.0)]
    assert [h.chunk_id for _, h in fuse(cands, alpha=0.1)] == [3, 2, 1]


def test_fuse_alpha_zero_is_pure_cosine_and_missing_lexical_is_zero():
    cands = [rc(1, 0.5, None), rc(2, 0.7, 1.0)]
    assert [h.chunk_id for _, h in fuse(cands, alpha=0.0)] == [2, 1]
    assert fuse([rc(1, 0.5, None)], alpha=0.3)[0][0] == pytest.approx(0.5)


def test_fuse_ties_broken_by_chunk_id():
    assert [h.chunk_id for _, h in fuse([rc(9, 0.5), rc(4, 0.5)], 0.1)] == [4, 9]


# ---------- Retriever ----------
def test_dense_mode_filters_per_chunk_and_reports_top_score():
    store = StubStore(dense=[rc(1, 0.8), rc(2, 0.4)])
    res = Retriever(StubEmbedder(), store, min_score=0.5).retrieve("q", 5)
    assert [h.chunk_id for h in res.results] == [1]
    assert res.top_score == 0.8
    assert store.calls == [("dense", 5)]


def test_hybrid_blocks_query_when_best_cosine_below_threshold():
    # khớp từ khóa mạnh không "cứu" được câu hỏi ngoài phạm vi
    store = StubStore(hybrid=[rc(1, 0.45, 1.0), rc(2, 0.30, 1.0)])
    res = Retriever(StubEmbedder(), store, min_score=0.5, mode="hybrid", alpha=0.3).retrieve("q", 5)
    assert res.results == [] and res.top_score == 0.45


def test_hybrid_keeps_chunks_whose_fused_score_passes():
    cands = [rc(1, 0.70, 0.0), rc(2, 0.45, 0.8), rc(3, 0.40, 0.2)]
    res = Retriever(StubEmbedder(), StubStore(hybrid=cands), min_score=0.5, mode="hybrid",
                    alpha=0.1).retrieve("q", 5)
    # 2: 0.45 + 0.08 = 0.53 ≥ 0.5 → giữ; 3: 0.42 → bỏ
    assert [h.chunk_id for h in res.results] == [1, 2]
    assert res.results[1].score == 0.45  # score trả ra vẫn là cosine


def test_hybrid_passes_candidates_and_truncates_to_top_k():
    store = StubStore(hybrid=[rc(i, 0.9 - i * 0.01) for i in range(1, 30)])
    res = Retriever(StubEmbedder(), store, min_score=0.0, mode="hybrid", candidates=20).retrieve("q", 3)
    assert [h.chunk_id for h in res.results] == [1, 2, 3]
    assert store.calls == [("hybrid", 20)]


def test_hybrid_rare_identifier_rises_with_memory_store(fake_embedder, memory_store):
    # Từ phổ biến (chạy/lúc/mấy/giờ) có ở hầu hết chunk → idf thấp; recon.daily chỉ ở 1 chunk.
    # FakeEmbedder là bag-of-words nên chênh cosine lớn → dùng alpha lớn để kiểm tra cơ chế.
    texts = [
        "chạy lúc mấy giờ",
        "job recon.daily chạy lúc 01:00",
        "mấy giờ chạy",
        "lúc mấy giờ",
        "chạy lúc mấy giờ nghỉ",
    ]
    chunks = [Chunk(i, t, (), 0, len(t), 5) for i, t in enumerate(texts)]
    memory_store.replace_document(Document("a.md", "x", "h"), chunks, fake_embedder.embed(texts))
    q = "recon.daily chạy lúc mấy giờ"
    dense = Retriever(fake_embedder, memory_store, min_score=-1.0).retrieve(q, 3)
    hybrid = Retriever(fake_embedder, memory_store, min_score=-1.0, mode="hybrid", alpha=1.5).retrieve(q, 3)
    assert dense.results[0].text == texts[0]  # dense: chunk trùng nhiều từ phổ biến nhất thắng
    assert hybrid.results[0].text == texts[1]  # hybrid: từ hiếm recon.daily có idf cao


def test_empty_query_rejected():
    with pytest.raises(ValueError):
        Retriever(StubEmbedder(), StubStore()).retrieve("  ", 5)
