"""Prompt có context đánh số [1]..[n] và kiểm tra trích dẫn (Docs/01-architecture.md §8)."""

import re

from rag_engine.models import RetrievedChunk

NOT_FOUND_MESSAGE = "Không tìm thấy thông tin trong tài liệu."

# Không có quy tắc "nếu không biết thì nói không tìm thấy": câu ngoài phạm vi đã bị chặn bởi
# min_score trước khi gọi LLM (Orchestrator). Với qwen2.5:3b, quy tắc đó khiến model từ chối cả câu
# trả lời được (đo ngày 2026-11-02: 9/10 → 12/12 khi bỏ). Xem Docs/02-milestones.md M3.
SYSTEM_PROMPT = """Bạn là trợ lý trả lời câu hỏi dựa trên tài liệu được cung cấp.
Quy tắc:
- Chỉ sử dụng thông tin trong phần CONTEXT. Không dùng kiến thức bên ngoài, không suy đoán.
- Các đoạn CONTEXT có thể khác ngôn ngữ với câu hỏi.
- BẮT BUỘC: cuối mỗi câu trả lời phải có trích dẫn dạng [n], n là số thứ tự đoạn context đã dùng.
- Nếu CONTEXT chỉ trả lời được một phần, trả lời phần đó và nói rõ phần còn thiếu.
- Trả lời ngắn gọn, chính xác, bằng cùng ngôn ngữ với câu hỏi.

Ví dụ:
CONTEXT:
[1] (source: a.md › Giờ làm)
Văn phòng mở cửa từ 8:00 đến 17:00.
[2] (source: b.md › Gửi xe)
Gửi xe máy miễn phí.
QUESTION: Văn phòng mở cửa lúc mấy giờ?
TRẢ LỜI: Văn phòng mở cửa từ 8:00 đến 17:00 [1]."""

REMINDER = "Trả lời câu hỏi trên, nhớ ghi trích dẫn [n] sau mỗi câu."

_CITATION_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def format_source(chunk: RetrievedChunk) -> str:
    return chunk.source + (" › " + " › ".join(chunk.heading_path) if chunk.heading_path else "")


class PromptBuilder:
    def __init__(self, context_max_chars: int = 8000, system_prompt: str = SYSTEM_PROMPT) -> None:
        self.context_max_chars = context_max_chars
        self.system_prompt = system_prompt

    def select_context(self, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """Giữ chunk theo thứ tự score cho tới khi chạm budget ký tự (luôn giữ ít nhất 1)."""
        selected: list[RetrievedChunk] = []
        used = 0
        for c in chunks:
            if selected and used + len(c.text) > self.context_max_chars:
                break
            selected.append(c)
            used += len(c.text)
        return selected

    def build(self, question: str, chunks: list[RetrievedChunk]) -> list[dict]:
        blocks = [f"[{i}] (source: {format_source(c)})\n{c.text}" for i, c in enumerate(chunks, 1)]
        user = "CONTEXT:\n" + "\n\n".join(blocks) + f"\n\nQUESTION: {question}\n\n{REMINDER}"
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user},
        ]


def extract_citations(text: str, n_sources: int) -> tuple[list[int], list[int]]:
    """Trả về (trích dẫn hợp lệ, trích dẫn không hợp lệ), mỗi loại không trùng, theo thứ tự xuất hiện."""
    valid: list[int] = []
    invalid: list[int] = []
    for m in _CITATION_RE.finditer(text):
        for part in m.group(1).split(","):
            n = int(part)
            bucket = valid if 1 <= n <= n_sources else invalid
            if n not in bucket:
                bucket.append(n)
    return valid, invalid
