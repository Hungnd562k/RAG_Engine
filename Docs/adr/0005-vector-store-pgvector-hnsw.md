# ADR-0005: PostgreSQL + pgvector với index HNSW cosine

- **Trạng thái:** Accepted
- **Ngày:** 2026-11-02

## Bối cảnh
Cần nơi lưu document, chunk, metadata và vector; tìm kiếm láng giềng gần; về sau cần lọc metadata, full-text (hybrid) và transaction khi re-ingest. Đã chọn trước PGVector.

## Quyết định
- PostgreSQL 16 với extension **pgvector** (image `pgvector/pgvector:pg16`).
- Bảng `documents`, `chunks` (cột `embedding vector(1024)`), `engine_meta` — chi tiết ở [03-data-model.md](../03-data-model.md).
- Index **HNSW** với `vector_cosine_ops` (`m=16`, `ef_construction=64`), `hnsw.ef_search` mặc định 40, chỉnh được.
- Re-ingest một document trong một transaction (delete chunks cũ + insert mới).

## Các phương án đã cân nhắc
| Phương án | Ưu | Nhược |
|-----------|----|-------|
| pgvector HNSW (chọn) | Recall/tốc độ tốt, build được trên bảng rỗng, không cần retrain | Build index chậm hơn, tốn RAM hơn IVFFlat |
| pgvector IVFFlat | Build nhanh, nhẹ | Cần dữ liệu trước khi build, recall phụ thuộc `lists/probes` |
| Không index (exact scan) | Chính xác tuyệt đối | Chậm khi > vài chục nghìn chunk |
| Qdrant / Milvus / Weaviate | Chuyên vector, hybrid sẵn | Thêm một hệ thống; metadata/transaction kém tiện bằng SQL |
| Chroma / FAISS | Đơn giản, nhúng được | Không phải DB đầy đủ, khó mở rộng thành service |

## Hệ quả
- Một DB duy nhất cho dữ liệu quan hệ + vector + (sau này) full-text `tsvector` → hybrid search không cần thêm hệ thống.
- Lọc metadata bằng SQL thường; lưu ý filter chọn lọc mạnh có thể làm HNSW trả ít hơn top-k (pgvector ≥ 0.8 có iterative scan để khắc phục).
- Dimension cố định trong schema: đổi model khác chiều cần migration + re-embed.
