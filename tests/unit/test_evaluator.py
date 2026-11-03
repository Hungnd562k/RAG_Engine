import math

import pytest

from rag_engine.evaluation.evaluator import (
    GoldenItem,
    evaluate,
    is_relevant,
    load_golden,
    percentile,
    source_matches,
)
from rag_engine.models import RetrievedChunk, SearchResult


class ScriptedRetriever:
    """Trả kết quả soạn sẵn theo câu hỏi."""

    def __init__(self, table: dict[str, list[RetrievedChunk]], min_score: float = 0.5) -> None:
        self.table = table
        self.min_score = min_score

    def retrieve(self, query, top_k, source_prefix=None, min_score=None):
        return SearchResult(self.table[query][:top_k], {"embed": 10.0, "search": 2.0, "total": 12.0})


def ch(source: str, score: float, heading: tuple[str, ...] = ()) -> RetrievedChunk:
    return RetrievedChunk(1, source, "text", heading, score)


def test_source_matches_exact_and_suffix():
    assert source_matches("samples/a.md", "samples/a.md")
    assert source_matches("/home/u/E-Wallet/docs/prd.md", "E-Wallet/docs/prd.md")
    assert not source_matches("/home/u/X-E-Wallet/docs/prd.md", "E-Wallet/docs/prd.md")
    assert not source_matches("samples/a.md.bak", "samples/a.md")


def test_is_relevant_checks_heading_case_insensitive():
    item = GoldenItem(id="x", question="q", expected_sources=["a.md"], expected_heading="nghỉ phép")
    assert is_relevant(ch("a.md", 0.9, ("Chính sách", "Nghỉ phép năm")), item)
    assert not is_relevant(ch("a.md", 0.9, ("Lương",)), item)
    assert not is_relevant(ch("b.md", 0.9, ("Nghỉ phép",)), item)


def test_golden_validation(tmp_path):
    with pytest.raises(ValueError):
        GoldenItem(id="x", question="q")  # trong phạm vi nhưng thiếu expected_sources
    f1 = tmp_path / "a.yaml"
    f1.write_text("- {id: a, question: q, expected_sources: [x.md]}\n", encoding="utf-8")
    f2 = tmp_path / "b.yaml"
    f2.write_text("- {id: a, question: q2, out_of_scope: true}\n", encoding="utf-8")
    assert len(load_golden(f1)) == 1
    with pytest.raises(ValueError, match="Trùng id"):
        load_golden(f1, f2)


@pytest.mark.parametrize(
    ("answer", "expected", "ok"),
    [
        ("The maximum is 50,000,000đ [5].", ["50.000.000"], True),
        ("Tối đa 50.000.000đ/lần", ["50,000,000"], True),
        ("Giá 99 000 VND", ["99,000"], True),
        ("Gói PRO giá 99.000", ["pro", "99000"], True),
        ("Ngưỡng 1.5 triệu", ["15"], False),  # không gộp số thập phân
        ("12 ngày phép", ["12 tháng"], False),
    ],
)
def test_contains_all_normalizes_numbers(answer, expected, ok):
    from rag_engine.evaluation.evaluator import contains_all

    assert contains_all(answer, expected) is ok


def test_percentile():
    assert percentile([], 95) == 0.0
    assert percentile([5, 1, 3], 50) == 3
    assert percentile(list(range(1, 101)), 95) == 95


def test_evaluate_metrics():
    items = [
        GoldenItem(id="r1", question="q1", expected_sources=["a.md"]),   # rank 1
        GoldenItem(id="r3", question="q2", expected_sources=["b.md"]),   # rank 3
        GoldenItem(id="miss", question="q3", expected_sources=["z.md"]),  # không có
        GoldenItem(id="oos1", question="o1", out_of_scope=True),          # bị chặn
        GoldenItem(id="oos2", question="o2", out_of_scope=True),          # lọt
    ]
    table = {
        "q1": [ch("a.md", 0.8), ch("b.md", 0.6)],
        "q2": [ch("x.md", 0.7), ch("y.md", 0.6), ch("b.md", 0.55)],
        "q3": [ch("x.md", 0.45)],
        "o1": [ch("x.md", 0.3)],
        "o2": [ch("x.md", 0.52)],
    }
    rep = evaluate(items, ScriptedRetriever(table, min_score=0.5), top_k=5)
    m = rep.metrics
    assert (rep.n_in_scope, rep.n_oos) == (3, 2)
    assert m["hit@1"] == pytest.approx(1 / 3)
    assert m["hit@3"] == pytest.approx(2 / 3)
    assert m["mrr"] == pytest.approx((1 + 1 / 3) / 3)
    assert m["in_scope_pass"] == pytest.approx(2 / 3)  # q3 top1 0.45 < 0.5 → bị chặn nhầm
    assert m["oos_block"] == pytest.approx(1 / 2)
    assert m["min_in_scope_top1"] == 0.45 and m["max_oos_top1"] == 0.52
    assert [r.id for r in rep.misses(1)] == ["r3", "miss"]


def test_evaluate_with_answer_uses_orchestrator():
    from rag_engine.models import Answer

    class FakeOrch:
        def ask(self, question, top_k, source_prefix=None, min_score=None):
            if question == "o1":
                return Answer("Không tìm thấy", [], [], [], {"total": 1.0}, answered=False)
            return Answer("12 ngày [1]", [ch("a.md", 0.8)], [1], [], {"generate": 100.0, "total": 110.0})

    items = [
        GoldenItem(id="a", question="q1", expected_sources=["a.md"], answer_contains=["12 ngày"]),
        GoldenItem(id="b", question="q1", expected_sources=["a.md"], answer_contains=["99"]),
        GoldenItem(id="o", question="o1", out_of_scope=True),
    ]
    table = {"q1": [ch("a.md", 0.8)], "o1": [ch("x.md", 0.2)]}
    m = evaluate(items, ScriptedRetriever(table), FakeOrch(), top_k=5).metrics
    assert m["answer_ok"] == 0.5 and m["citation_ok"] == 1.0 and m["oos_not_answered"] == 1.0
    assert m["generate_p50_ms"] == 100.0


def test_empty_oos_gives_nan():
    items = [GoldenItem(id="a", question="q1", expected_sources=["a.md"])]
    m = evaluate(items, ScriptedRetriever({"q1": [ch("a.md", 0.8)]})).metrics
    assert math.isnan(m["oos_block"])
