# 07 — Rủi ro & Câu hỏi mở

## 1. Rủi ro

| # | Rủi ro | Ảnh hưởng | Khả năng | Giảm thiểu |
|---|--------|-----------|----------|------------|
| R1 | Máy không có GPU → `qwen2.5:7b` sinh câu trả lời chậm (30–90s) | Trải nghiệm "thấy ngay" kém | Cao | Cho phép đổi `RAG_LLM_MODEL` sang `qwen2.5:3b-instruct`; `rag search` vẫn nhanh để kiểm chứng retrieval độc lập |
| R2 | Ollama chỉ trả dense vector của BGE-M3, mất sparse/ColBERT | Recall với từ khóa hiếm/mã số kém | Trung bình | Hybrid search bằng Postgres full-text ở M6 ([ADR-0004](adr/0004-embedding-bge-m3-via-ollama.md)) |
| R3 | Dimension lock-in (vector(1024)) | Đổi model phải re-embed toàn bộ | Thấp (MVP) | `engine_meta` phát hiện lệch; `rag reset` + ingest lại |
| R4 | Chunk size tính theo ký tự ước lượng, sai khác token thật với tiếng Việt | Chunk quá dài/ngắn | Trung bình | Đo bằng `rag eval`; cân nhắc tokenizer của bge-m3 (`transformers`) nếu cần chính xác |
| R5 | LLM bịa hoặc trích dẫn sai `[n]` | Mất tin cậy | Trung bình | Prompt nghiêm, temperature thấp, `min_score` short-circuit, validate citation hậu kỳ |
| R6 | Ollama thay đổi API/hành vi `/api/embed` (normalize, truncate) | Kết quả lệch | Thấp | Pin image version khi ổn định; integration test kiểm tra dim/cosine |
| R7 | Python 3.14 hệ thống thiếu wheel cho vài lib | Cài đặt lỗi | Trung bình | Pin 3.12 qua uv |
| R8 | Golden set nhỏ, chủ quan | Metrics không đại diện | Cao | Coi là smoke metric; mở rộng dần khi có dữ liệu thật |

## 2. Câu hỏi mở
| # | Câu hỏi | Cần trả lời trước |
|---|---------|-------------------|
| Q1 | Dữ liệu thật đầu tiên là gì (domain, ngôn ngữ, khối lượng)? Ảnh hưởng chunk size & golden set | M5 |
| Q2 | Engine sẽ được dùng như **thư viện** import hay **service** HTTP là chính? | M4 |
| Q3 | Có cần lưu lịch sử hội thoại (multi-turn, query rewriting)? | M6 |
| Q4 | Yêu cầu phân quyền theo tài liệu/người dùng? | Trước khi dùng dữ liệu nhạy cảm |
| Q5 | Có GPU không? Quyết định model LLM mặc định | M0 |
| Q6 | Ngưỡng `min_score` 0.35 có hợp với BGE-M3 trên dữ liệu thật? | M5 |
