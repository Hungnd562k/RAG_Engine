# 00 — Vision & Scope

## 1. Vấn đề
Cần một RAG engine dùng lại được cho nhiều ứng dụng: nạp tài liệu nội bộ, tìm kiếm ngữ nghĩa, và trả lời câu hỏi bằng LLM **có trích dẫn nguồn**. Trước khi tính tới scale/production, cần một bản **local tối giản** để hiểu rõ từng bước và kiểm chứng chất lượng.

## 2. Mục tiêu MVP (local)
1. Chạy hoàn toàn offline trên một máy: Postgres + pgvector, Ollama, app Python.
2. Nạp được thư mục tài liệu TXT/Markdown bằng **một lệnh**.
3. Tìm kiếm ngữ nghĩa (`rag search`) và hỏi đáp (`rag ask`) — kết quả hiển thị ngay trên terminal, kèm nguồn, score và thời gian từng bước.
4. Có REST API (FastAPI + Swagger UI) để tích hợp/thử qua trình duyệt.
5. Tự kiểm chứng được: bộ dữ liệu mẫu + golden set + lệnh `rag eval` in ra hit@k/MRR.

## 3. Phạm vi

### In-scope (M0–M5)
- Loader TXT/MD; chunker recursive, nhận biết heading Markdown.
- Embedding BGE-M3 (dense, 1024 chiều) qua Ollama.
- Lưu trữ + tìm kiếm vector bằng pgvector (HNSW, cosine).
- Ingest idempotent (chạy lại không nhân bản; file đổi nội dung thì thay thế).
- Generation qua Ollama với prompt bắt buộc trích dẫn `[n]`.
- CLI: `health`, `ingest`, `search`, `ask`, `stats`, `reset`, `eval`.
- API: `/health`, `/ingest`, `/search`, `/ask`.
- Unit test + integration test + retrieval evaluation.

### Out-of-scope (để M6+)
- PDF, DOCX, HTML, OCR.
- Hybrid search (keyword + vector), reranker.
- Multi-tenant, auth, phân quyền tài liệu.
- Streaming response, UI chat.
- Ingest bất đồng bộ / queue, scale ngang, deploy cloud.
- Đánh giá answer tự động bằng LLM-as-judge.

## 4. Tiêu chí thành công
| Tiêu chí | Ngưỡng MVP |
|----------|------------|
| Dựng môi trường từ zero theo [05-local-setup.md](05-local-setup.md) | < 30 phút (không tính thời gian tải model) |
| Ingest bộ `samples/` (~10–20 file) | < 1 phút trên CPU |
| `rag search` latency (không tính lần load model đầu) | < 1 giây |
| Retrieval hit@5 trên golden set | ≥ 0.8 |
| `rag ask` trả lời có trích dẫn hợp lệ | 100% câu trong golden set có ≥ 1 citation trỏ tới nguồn tồn tại |
| Câu hỏi ngoài phạm vi | Trả lời "không tìm thấy thông tin" thay vì bịa |

## 5. Người dùng mục tiêu
- Developer (chính chủ repo) để học, thử nghiệm, đo đạc.
- Về sau: các ứng dụng khác gọi engine qua API hoặc import như thư viện.

## 6. Nguyên tắc thiết kế
- **Minh bạch hơn tiện lợi:** không dùng framework RAG nặng; mỗi bước là module nhỏ, đọc được ([ADR-0003](adr/0003-no-rag-framework.md)).
- **Thay thế được:** Embedder, VectorStore, LLM đều đứng sau interface (Protocol).
- **Quan sát được:** mọi lệnh in timing và nguồn; log có cấu trúc.
- **Kiểm chứng được:** mỗi milestone có Definition of Done bằng lệnh cụ thể.
