# ADR-0004: Embedding BGE-M3 (dense) qua Ollama

- **Trạng thái:** Accepted
- **Ngày:** 2026-11-02

## Bối cảnh
Tài liệu dự kiến có cả tiếng Việt và tiếng Anh. Cần embedding model đa ngôn ngữ, chạy local, context dài. BGE-M3 (BAAI) hỗ trợ 100+ ngôn ngữ, input tới 8192 token, cho vector 1024 chiều, chất lượng retrieval tốt trên MIRACL/MKQA. Ollama cung cấp sẵn model `bge-m3` qua HTTP.

## Quyết định
- Dùng `bge-m3` qua Ollama endpoint `POST /api/embed` (batch `input: [...]`), batch size mặc định 16.
- Chỉ dùng **dense vector 1024 chiều**; lưu vào `vector(1024)`; so sánh bằng cosine.
- Không thêm instruction prefix cho query/passage (BGE-M3 không yêu cầu, khác BGE v1.5).
- Ghi `embed_model`, `embed_dim` vào bảng `engine_meta`; lệch cấu hình → báo lỗi, buộc reset.
- Embedder đứng sau `Embedder` Protocol để thay model khác về sau.

## Các phương án đã cân nhắc
| Phương án | Ưu | Nhược |
|-----------|----|-------|
| BGE-M3 qua Ollama (chọn) | Đa ngôn ngữ tốt, cùng runtime với LLM, không cần PyTorch trong app | Ollama chỉ expose dense, không có sparse/ColBERT |
| BGE-M3 qua `FlagEmbedding`/`sentence-transformers` | Có đủ dense + sparse + multi-vector | Kéo PyTorch (~GB) vào app, phức tạp hơn cho MVP |
| `nomic-embed-text` | Nhẹ, nhanh | Yếu hơn với tiếng Việt |
| `multilingual-e5-large` | Đa ngôn ngữ tốt | Cần prefix `query:`/`passage:`, context 512 token |
| API thương mại (OpenAI, Voyage…) | Chất lượng cao | Không offline, tốn phí, dữ liệu ra ngoài |

## Hệ quả
- **Tích cực:** app Python nhẹ; truy vấn cross-lingual hoạt động.
- **Tiêu cực:** mất lợi thế lexical matching của sparse vector → từ khóa hiếm/mã số có thể bị bỏ lỡ. Bù bằng hybrid search Postgres full-text ở M6.
- Lần gọi đầu chậm do Ollama load model; cân nhắc `OLLAMA_KEEP_ALIVE`.
- Cần xác minh trong integration test rằng vector trả về đã chuẩn hóa (hoặc chuẩn hóa ở client) — với cosine distance thì không bắt buộc nhưng giúp so sánh nhất quán.

## Tham khảo
- BGE-M3 paper: arXiv 2402.03216
- Ollama API: `/api/embed`
