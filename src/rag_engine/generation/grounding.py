"""Hậu kiểm con số: mọi con số trong câu trả lời phải có trong context hoặc câu hỏi (R11).

Không chặn câu trả lời, chỉ gắn cờ. Model 3B hay bịa con số (giờ chạy, hạn mức, số ngày) khi
retrieval trượt nhưng chunk sai vẫn vượt `min_score`; câu trả lời bịa vẫn có trích dẫn [n] nên
không thể phát hiện bằng kiểm tra trích dẫn.

So khớp theo *dạng chuẩn hóa*, không theo chuỗi:
- Tách theo dấu phân cách `. , : /` rồi bỏ số 0 ở đầu mỗi phần: `08:00` ≡ `8:00`,
  `1.000.000` ≡ `1,000,000`.
- Số có nhóm hàng nghìn (`1.000.000`, `1,000,000`) còn khớp dạng viết liền `1000000`.
- Thập phân không khớp lẫn với số nguyên: `1.5` ≠ `15`.
"""

import re

from rag_engine.generation.prompt import _CITATION_RE, format_source
from rag_engine.models import RetrievedChunk

# Số đứng riêng, không dính chữ: bắt "12", "01:00", "1.000.000", "15%", "30/04/2026",
# bỏ qua "2FA", "SHA256", "bge-m3", "qwen2.5", "M6a".
_NUMBER_RE = re.compile(r"(?<![\w.,:/])\d+(?:[.,:/]\d+)*(?![\w])")
_THOUSANDS_RE = re.compile(r"\d{1,3}(?:([.,])\d{3})(?:\1\d{3})*")
# Số thứ tự đầu dòng của danh sách ("1. ", "2) ") không phải dữ kiện.
_LIST_MARKER_RE = re.compile(r"^(\s*)\d+[.)](?=\s)", re.MULTILINE)


def _forms(token: str) -> set[tuple[int, ...]]:
    """Các dạng chuẩn hóa của một token số."""
    forms = {tuple(int(p) for p in re.split(r"[.,:/]", token))}
    if _THOUSANDS_RE.fullmatch(token):
        forms.add((int(re.sub(r"[.,]", "", token)),))
    return forms


def extract_numbers(text: str) -> list[str]:
    """Các token số trong text (bỏ trích dẫn [n] và số thứ tự danh sách), không trùng, theo thứ tự."""
    text = _LIST_MARKER_RE.sub(r"\1", _CITATION_RE.sub(" ", text))
    seen: list[str] = []
    for m in _NUMBER_RE.finditer(text):
        if m.group() not in seen:
            seen.append(m.group())
    return seen


def unsupported_numbers(answer: str, sources: list[RetrievedChunk], question: str = "") -> list[str]:
    """Các con số trong `answer` không xuất hiện (sau chuẩn hóa) trong context hoặc câu hỏi."""
    corpus = [question] + [f"{format_source(c)}\n{c.text}" for c in sources]
    known: set[tuple[int, ...]] = set()
    for text in corpus:
        for tok in extract_numbers(text):
            known |= _forms(tok)
    return [tok for tok in extract_numbers(answer) if not (_forms(tok) & known)]
