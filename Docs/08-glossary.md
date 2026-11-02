# 08 — Glossary

| Thuật ngữ | Giải thích |
|-----------|-----------|
| **RAG** (Retrieval-Augmented Generation) | Kỹ thuật lấy (retrieve) các đoạn tài liệu liên quan rồi đưa vào prompt để LLM sinh câu trả lời dựa trên đó. |
| **Document** | Một file nguồn đã nạp (TXT/MD), định danh bằng `source`. |
| **Chunk** | Một đoạn con của document, đơn vị được embed và tìm kiếm. |
| **Chunk overlap** | Phần nội dung lặp lại giữa hai chunk liền kề để không mất ngữ cảnh ở ranh giới. |
| **heading_path** | Chuỗi tiêu đề Markdown chứa chunk, vd. `["Nghỉ phép", "Nghỉ phép năm"]`. |
| **Embedding** | Vector số thực biểu diễn ngữ nghĩa của văn bản. BGE-M3 cho vector 1024 chiều. |
| **Dense / Sparse / Multi-vector** | Ba kiểu biểu diễn BGE-M3 hỗ trợ. Dense = 1 vector/đoạn (Ollama hỗ trợ); sparse = trọng số theo từ (như BM25); multi-vector = ColBERT, mỗi token một vector. |
| **Cosine similarity** | Độ giống giữa hai vector, 1 = cùng hướng. pgvector `<=>` trả cosine *distance* = 1 − similarity. |
| **pgvector** | Extension PostgreSQL cho kiểu `vector` và tìm kiếm láng giềng gần. |
| **HNSW** | Hierarchical Navigable Small World — index tìm kiếm láng giềng gần xấp xỉ (ANN), nhanh, recall cao. Tham số: `m`, `ef_construction`, `ef_search`. |
| **IVFFlat** | Index ANN khác của pgvector, dựa trên phân cụm; cần dữ liệu trước khi build. |
| **ANN** | Approximate Nearest Neighbor — tìm láng giềng gần xấp xỉ, đổi một chút độ chính xác lấy tốc độ. |
| **top-k** | Số kết quả gần nhất được lấy về. |
| **min_score** | Ngưỡng similarity tối thiểu; dưới ngưỡng coi như không liên quan. |
| **Citation** | Tham chiếu `[n]` trong câu trả lời trỏ tới chunk nguồn thứ n. |
| **Golden set** | Bộ câu hỏi kèm đáp án/nguồn kỳ vọng dùng để đánh giá. |
| **hit@k** | Tỉ lệ câu hỏi có kết quả đúng nằm trong top-k. |
| **MRR** | Mean Reciprocal Rank — trung bình 1/(thứ hạng kết quả đúng đầu tiên). |
| **Hybrid search** | Kết hợp tìm kiếm từ khóa (BM25/full-text) và vector. |
| **RRF** | Reciprocal Rank Fusion — trộn nhiều danh sách xếp hạng bằng tổng `1/(k + rank)`. |
| **Reranker** | Mô hình cross-encoder chấm lại điểm cặp (query, chunk) để sắp xếp chính xác hơn. |
| **Idempotent ingest** | Nạp lại cùng dữ liệu không làm thay đổi/nhân bản kết quả. |
| **Ollama** | Runtime chạy LLM/embedding model local, cung cấp HTTP API (`/api/embed`, `/api/chat`). |
| **ADR** | Architecture Decision Record — ghi lại một quyết định kiến trúc, bối cảnh và hệ quả. |
| **DoD** | Definition of Done — điều kiện để coi một milestone là hoàn thành. |
