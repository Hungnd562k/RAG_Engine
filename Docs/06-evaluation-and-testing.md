# 06 — Evaluation & Testing

## 1. Kim tự tháp kiểm chứng

| Tầng | Công cụ | Cần infra? | Khi chạy |
|------|---------|-----------|----------|
| Unit tests | `pytest -m "not integration"` | Không (fake Embedder/LLM/Store) | Mọi thay đổi |
| Integration tests | `pytest -m integration` | Compose đang chạy | Trước khi đóng milestone |
| Retrieval eval | `rag eval eval/golden.yaml` | Có | Khi đổi chunking/model/ngưỡng |
| Answer eval | `rag eval eval/golden.yaml --with-answer` | Có (LLM) | Khi đổi prompt/LLM |
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

## 6. Baseline

Cách chạy: `uv run rag eval eval/golden.yaml [--with-answer]`. Báo cáo JSON ghi ra `eval/report-<time>.json` (gitignored). Có thể truyền nhiều file golden; bộ golden riêng tư đặt ở `eval/local/` (gitignored).

### 6.1 Retrieval
| Ngày | Golden set | Cấu hình | hit@1 | hit@3 | hit@5 | MRR | OOS block | search p50 / p95 |
|------|-----------|----------|-------|-------|-------|-----|-----------|------------------|
| 2026-11-02 (sơ bộ, M2) | 6 câu + 8 OOS, script tạm | bge-m3, chunk=512, overlap=64, min_score=0.50, TXT tách đoạn | 1.00 | 1.00 | 1.00 | 1.00 | 8/8 | ~3 ms |
| **2026-11-03 (M5, chính thức)** | `eval/golden.yaml`: 19 + 6 OOS | như trên, top_k=5 | **1.00** | **1.00** | **1.00** | **1.00** | **6/6** | 4 / 6 ms (embed p50 45 ms) |
| 2026-11-03 (M5, đối chứng) | E-Wallet (riêng tư): 20 + 4 OOS | như trên | 0.90 | 0.90 | 0.90 | 0.90 | 3/4 | 5 / 6 ms |
| 2026-11-03 (M6a, thử, **loại**) | samples / E-Wallet | hybrid RRF (k=60), `ts_rank_cd` | 0.737 / 0.90 | 0.895 / 0.95 | 0.947 / 0.95 | 0.820 / 0.925 | 6/6 / 3/4 | 13 / 16 ms |
| **2026-11-03 (M6a, hybrid)** | `eval/golden.yaml` | `RAG_SEARCH_MODE=hybrid`, α=0.15, candidates=20 ([ADR-0010](adr/0010-hybrid-search-score-fusion.md)) | **1.00** | **1.00** | **1.00** | **1.00** | **6/6** | ~13 ms |
| **2026-11-03 (M6a, hybrid)** | E-Wallet | như trên | **0.95** | **0.95** | **0.95** | **0.95** | 3/4 | ~15 ms |
| 2026-11-03 (M6a.2, hybrid + rerank 20) | E-Wallet* | hybrid → `bge-reranker-v2-m3` (TEI GPU) | 0.95 | 0.95 | 0.95 | 0.950 | 3/4 | 18 ms + rerank ~0.77 s |
| 2026-11-03 (M6a.2, hybrid + rerank 40) | E-Wallet* | như trên, rerank toàn bộ hợp ứng viên | 0.95 | 1.00 | 1.00 | 0.963 | 3/4 | rerank ~1.7 s |
| **2026-11-03 (M6a.2, dense + rerank 20)** | `eval/golden.yaml` | `RAG_RERANK_ENABLED=true`, `RAG_SEARCH_MODE=auto` → dense ([ADR-0011](adr/0011-reranker-tei-service.md)) | **1.00** | **1.00** | **1.00** | **1.00** | **6/6** | 6 ms + rerank ~0.5 s |
| **2026-11-03 (M6a.2, dense + rerank 20)** | E-Wallet* | như trên | **0.95** | **1.00** | **1.00** | **0.967** | 3/4 | 7 ms + rerank ~0.7 s |

\* Từ M6a.2, golden E-Wallet e09 bổ sung `chuan-hoa/B-bo-tai-lieu-chuan-hoa.md` (bảng thuật ngữ "QR động … hết hạn sau 15 phút" cũng chứa đáp án; reranker chọn chunk này lên hạng 1). Với golden đã sửa, hybrid không rerank vẫn là 0.95 / 0.95 / 0.95 / 0.950.

Bản TEI chạy CPU: điểm rerank giống bản GPU nhưng mất ~12 s cho 20 ứng viên, nên không dùng.

Quét α (hybrid, cosine + α × độ phủ IDF), hit@1 samples / E-Wallet: 0 → 1.00 / 0.90 · 0.05 → 1.00 / 0.90 · 0.10 → 1.00 / 0.90 (e14 hạng 3) · **0.15 → 1.00 / 0.95** · 0.20 → 1.00 / 0.90 (e09 tụt xuống hạng 2) · 0.30 → 0.947 / 0.90 (q04 khác ngôn ngữ tụt xuống hạng 2). Vùng tốt hẹp (0.1–0.2) → đo lại khi golden set lớn hơn.

Khoảng cách điểm top-1: trên `samples/`, câu trong phạm vi thấp nhất là 0.583 và câu OOS cao nhất là 0.479, nên ngưỡng 0.50 tách sạch hai nhóm. Trên E-Wallet, hai nhóm chồng lên nhau (trong phạm vi thấp nhất 0.551, OOS cao nhất 0.516).

### 6.2 Answer (`--with-answer`, qwen2.5:3b-instruct, GPU)
| Ngày | Golden set | Prompt | answer_ok | citation_ok | OOS không trả lời | generate p50 |
|------|-----------|--------|-----------|-------------|-------------------|--------------|
| **2026-11-03 (chính thức)** | `eval/golden.yaml` | M3 (few-shot "giờ mở cửa") | **0.947** | **1.00** | 6/6 | ~1.5 s |
| 2026-11-03 | E-Wallet | M3 | 0.95 | 0.90 | 3/4 | ~2.4 s |
| 2026-11-03 (thử, **loại**) | `eval/golden.yaml` / E-Wallet | Few-shot trung tính (không có số) | 0.947 / 0.90 | 0.895 / 0.85 | — | — |
| 2026-11-03 (thử, **loại**) | E-Wallet | Quy tắc "[1] hoặc [2]" | 0.80 | — | — | — |
| **2026-11-03 (M6a, hybrid)** | `eval/golden.yaml` | M3 | **1.00** | **1.00** | 6/6 | — |
| **2026-11-03 (M6a, hybrid)** | E-Wallet | M3 | 0.95 | 0.90 | 3/4 | — |
| **2026-11-03 (M6a.2, dense + rerank)** | `eval/golden.yaml` | M3 | **1.00** | **1.00** | 6/6 | ~3.4 s (LLM bị đẩy sang CPU) |
| **2026-11-03 (M6a.2, dense + rerank)** | E-Wallet | M3 | **1.00** | **1.00** | 3/4 | ~6.6 s (LLM 74% CPU) |

Ở M6a, e14 trả lời đúng "01:00" thay vì bịa giờ (R11). Các câu còn sai:
- e06: diễn giải sai quy tắc thiết bị.
- e14: trích `[6]` trong khi chỉ có 5 nguồn.
- e17: in `[n.5]`, tức lỗi `[n.x]` xuất hiện cả với prompt M3, không chỉ với few-shot trung tính.
- eo2: OOS lọt (0.516).

- `answer_ok`: mọi chuỗi `answer_contains` đều xuất hiện trong câu trả lời. Trước khi so khớp, dấu phân cách số được chuẩn hóa (`1.200.000` ≡ `1,200,000` ≡ `1200000`).
- `citation_ok`: có ít nhất một `[n]` hợp lệ và không có `[n]` nào trỏ ra ngoài context.

### 6.3 Tinh chỉnh tham số (M5)
- **chunk_size** 256 / 384 / 512 / 768 (overlap tương ứng) cho hit@1 như nhau (0.949 trên hai bộ gộp). Các tài liệu đều có heading rõ, nên chunk thường dừng ở biên section trước khi chạm `chunk_size`. **Giữ 512/64.**
- **min_score**: giữ **0.50**. Câu OOS lọt duy nhất là E-Wallet "cài Kubernetes trên Ubuntu" (0.516), vì trùng từ vựng với tài liệu quy trình review/CI. Nâng ngưỡng lên 0.52 sẽ rất sát câu trong phạm vi thấp nhất (0.551). Ở câu này LLM vẫn trả lời "không tìm thấy", nên lớp bảo vệ thứ hai vẫn có tác dụng.
- **Prompt few-shot**: ví dụ mẫu bị nghi rò rỉ, vì khi retrieval trượt (E-Wallet e14, giờ chạy `recon.daily`) model trả lời "8:00 đến 17:00", trùng số trong ví dụ. Đã thử thay bằng ví dụ trung tính: model vẫn bịa ("8 giờ"), đồng thời phát sinh lỗi mới là in nguyên văn `[n.1]` / `[n 3]`. **Giữ prompt M3.** Gốc rễ là retrieval trượt, nên hướng xử lý nằm ở retrieval (hybrid/reranker, M6) và ở việc kiểm tra con số trong câu trả lời có xuất hiện trong context hay không (xem R11 trong [07](07-risks-and-open-questions.md)).

Mỗi lần thay đổi chunking/model/ngưỡng/prompt: thêm một dòng mới vào các bảng trên để so sánh.
