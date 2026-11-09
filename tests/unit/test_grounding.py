from rag_engine.generation.grounding import extract_numbers, unsupported_numbers
from rag_engine.models import RetrievedChunk


def ctx(*texts: str) -> list[RetrievedChunk]:
    return [RetrievedChunk(i, f"docs/{i}.md", t, ("H",), 0.8) for i, t in enumerate(texts, 1)]


def test_extract_skips_citations_list_markers_and_identifiers():
    text = "1. Dùng bge-m3 và qwen2.5 với 2FA, SHA256 [1].\n2) Hạn mức 5.000.000 đồng [2, 3]."
    assert extract_numbers(text) == ["5.000.000"]


def test_extract_keeps_times_dates_percent():
    assert extract_numbers("Chạy lúc 01:00 ngày 30/04/2026, giảm 15%.") == ["01:00", "30/04/2026", "15"]


def test_grounded_answer_has_no_unsupported():
    sources = ctx("Job recon.daily chạy lúc 01:00 mỗi ngày.", "Hạn mức 1.000.000 VND.")
    assert unsupported_numbers("Job chạy lúc 1:00 [1], hạn mức 1,000,000 [2].", sources) == []
    assert unsupported_numbers("Hạn mức 1000000 VND [2].", sources) == []


def test_hallucinated_number_is_flagged():
    # E-Wallet e14 trước hybrid: chunk sai, model bịa giờ hành chính
    sources = ctx("Văn phòng làm việc theo giờ hành chính.", "Job recon.daily đối soát giao dịch.")
    assert unsupported_numbers("Job recon.daily chạy lúc 8:00 [2].", sources) == ["8:00"]


def test_decimal_does_not_match_integer():
    assert unsupported_numbers("Tỉ lệ 1.5% [1].", ctx("Tỉ lệ 15%.")) == ["1.5"]


def test_numbers_from_question_and_heading_count():
    sources = [RetrievedChunk(1, "docs/a.md", "Thời hạn lưu trữ.", ("Điều 12",), 0.8)]
    assert unsupported_numbers("Theo Điều 12, lưu 10 năm [1].", sources, "Lưu 10 năm đúng không?") == []
