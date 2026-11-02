# 06 — Evaluation & Testing

## 1. Kim tự tháp kiểm chứng

| Tầng | Công cụ | Cần infra? | Khi chạy |
|------|---------|-----------|----------|
| Unit tests | `pytest -m "not integration"` | Không (fake Embedder/LLM/Store) | Mọi thay đổi |
| Integration tests | `pytest -m integration` | Compose đang chạy | Trước khi đóng milestone |
| Retrieval eval | `rag eval eval/golden.yaml` | Có | Khi đổi chunking/model/ngưỡng |
| Kiểm tra thủ công | `rag search`, `rag ask --show-context` | Có | Hằng ngày |

## 2. Unit tests (mục tiêu)
- **Chunker:** text rỗng; text ngắn hơn chunk_size → 1 chunk; overlap đúng; không cắt giữa từ; `heading_path` đúng với MD lồng nhau; `char_start/end` khớp text gốc.
- **Loader:** bỏ qua file không hỗ trợ; hash ổn định; encoding UTF-8 có BOM.
- **Prompt Builder:** đánh số `[n]` đúng thứ tự; cắt theo `context_max_chars`; không có chunk → không gọi LLM.
- **Orchestrator:** dùng `FakeEmbedder` (vector xác định từ hash) + `InMemoryStore` + `FakeLLM` để test luồng và timings.
- **Config:** giá trị mặc định, override bằng env.

## 3. Integration tests
- Store: `replace_document` idempotent; cascade delete; search trả đúng thứ tự với vector dựng sẵn.
- Embedder: dim = 1024; hai câu đồng nghĩa có cosine > hai câu không liên quan.
- LLM: trả về chuỗi không rỗng với prompt ngắn (timeout dài).
- Dùng database riêng `rag_test` (tạo/xóa trong fixture) để không đụng dữ liệu dev.

## 4. Golden set

`eval/golden.yaml`:
```yaml
- id: q01
  question: "Nhân viên được nghỉ phép bao nhiêu ngày mỗi năm?"
  expected_sources: ["samples/hr-policy.md"]
  expected_heading: "Nghỉ phép năm"     # tùy chọn, khớp chặt hơn
  answer_contains: ["12 ngày"]           # dùng cho kiểm tra answer (tùy chọn)
- id: q20
  question: "Giá bitcoin hôm nay là bao nhiêu?"
  out_of_scope: true                     # kỳ vọng: không có chunk đạt min_score / trả lời "không tìm thấy"
```
Nguyên tắc: ≥ 20 câu; có câu hỏi diễn đạt khác từ ngữ tài liệu (paraphrase); có câu cross-lingual (hỏi VI, tài liệu EN); có 2–3 câu ngoài phạm vi.

## 5. Metrics
- **hit@k:** tỉ lệ câu có ít nhất một chunk thuộc `expected_sources` (và `expected_heading` nếu có) trong top-k.
- **MRR:** trung bình `1/rank` của kết quả đúng đầu tiên (0 nếu không có trong top-k).
- **Out-of-scope precision:** tỉ lệ câu `out_of_scope` bị chặn đúng bởi `min_score`.
- **Latency:** p50/p95 cho embed, search (và generate nếu chạy `--with-answer`).
- **Answer check (bán tự động):** `answer_contains` có xuất hiện; mọi `[n]` hợp lệ.

## 6. Baseline (điền khi xong M5)
| Ngày | Cấu hình | hit@1 | hit@3 | hit@5 | MRR | OOS precision | search p50 |
|------|----------|-------|-------|-------|-----|---------------|-----------|
| — | chunk=512, overlap=64, min_score=0.35 | | | | | | |

Mỗi lần thay đổi chunking/model/ngưỡng: thêm một dòng mới để so sánh.
