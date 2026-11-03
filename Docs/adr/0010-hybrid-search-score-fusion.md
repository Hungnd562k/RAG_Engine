# ADR-0010: Hybrid search gộp theo điểm (cosine + α × độ phủ IDF), không dùng RRF

- **Trạng thái:** Accepted
- **Ngày:** 2026-11-03

## Bối cảnh
Dense search thuần (BGE-M3 qua Ollama, [ADR-0004](0004-embedding-bge-m3-via-ollama.md)) bỏ sót các câu hỏi chứa mã hoặc định danh hiếm (R2). Ở M5 có một ví dụ cụ thể: câu E-Wallet e14 "Job đối soát `recon.daily` chạy lúc mấy giờ?" không có chunk đúng trong top 5. Dù vậy, chunk sai vẫn vượt `min_score`, nên model 3B bịa giờ chạy (R11).

Kế hoạch ban đầu ([01 §9](../01-architecture.md#9-điểm-mở-rộng-m6)) là thêm cột `tsvector` rồi gộp hai danh sách bằng Reciprocal Rank Fusion. Cả hai cách RRF đã đo đều kém:
- **RRF với `ts_rank_cd`:** bộ `samples` tụt hit@1 từ 1.00 xuống 0.737. Nguyên nhân là `ts_rank_cd` không có IDF, nên từ phổ biến ("là", "bao nhiêu", "của") khớp gần như mọi chunk, làm danh sách full-text gần như ngẫu nhiên.
- **RRF nói chung:** điểm RRF luôn cao hơn với chunk có mặt ở *cả hai* danh sách. Với câu hỏi khác ngôn ngữ tài liệu (q04: hỏi tiếng Việt, tài liệu tiếng Anh), full-text không thể khớp chunk đúng, nên chunk tiếng Việt "trông giống" vượt qua chunk tiếng Anh đang đứng hạng 1 dense với cosine 0.746. Đổi `k` hay hạ trọng số danh sách full-text không thay đổi được bất đẳng thức này.

## Quyết định
Chúng ta gộp theo **điểm**, trong một truy vấn SQL:
- **Ứng viên:** hợp của top-`RAG_HYBRID_CANDIDATES` (20) theo cosine và top-20 theo full-text.
- **Full-text:** cột `content_tsv` sinh tự động từ `to_tsvector('simple', content)`, có GIN index. Config `simple` giữ nguyên dấu tiếng Việt và mã định danh (`recon.daily`, `RAG_MIN_SCORE`).
- **Độ phủ từ khóa:** `lexical = Σ idf(từ của câu hỏi có trong chunk) / Σ idf(mọi từ của câu hỏi)`, trong đó `idf = ln(1 + (N − df + 0.5)/(df + 0.5))` (IDF kiểu BM25, chưa có tf và chuẩn hóa độ dài). df được đếm qua GIN index tại thời điểm truy vấn.
- **Xếp hạng:** `fused = cosine + α × lexical`, với **α = 0.15** (`RAG_HYBRID_ALPHA`).
- **Chặn câu ngoài phạm vi:** vẫn dùng **cosine cao nhất** so với `min_score`. Khớp từ khóa không "cứu" được câu ngoài phạm vi. Sau khi qua bước chặn, giữ các chunk có `fused ≥ min_score`. Trường `score` trả ra vẫn là cosine.
- `RAG_SEARCH_MODE=dense` cho phép quay lại dense thuần để so sánh.

## Các phương án đã cân nhắc
| Phương án | Ưu | Nhược |
|-----------|----|-------|
| Cosine + α × độ phủ IDF (chọn) | α nhỏ chỉ đảo thứ tự khi cosine sát nhau, nên câu khác ngôn ngữ vẫn an toàn. Một truy vấn SQL, không cần extension | Phụ thuộc thang cosine của BGE-M3 (dải 0.45–0.75). α phải đo lại khi đổi model. Đếm df mỗi truy vấn làm search p50 tăng từ ~5 ms lên ~15 ms |
| RRF (`ts_rank_cd`) | Không cần chuẩn hóa thang điểm, phổ biến | samples hit@1 0.737. Thua về cấu trúc với câu khác ngôn ngữ |
| RRF (xếp full-text theo IDF) | Như trên | Vẫn thua ở q04 (xem Bối cảnh) |
| BM25 thật (ParadeDB `pg_search`, `rank_bm25` trong Python) | Có tf và chuẩn hóa độ dài | Đổi image Postgres hoặc giữ index trong RAM. Chưa cần ở quy mô hiện tại |
| Tiếp tục dense thuần | Đơn giản | Không giải quyết được R2/R11 |

## Hệ quả
- **Tích cực:**
  - Bộ E-Wallet: hit@1 tăng từ 0.90 lên 0.95, e14 từ không có trong top 5 lên hạng 1.
  - Bộ `samples` giữ nguyên 1.00. Tỉ lệ chặn câu ngoài phạm vi không đổi.
  - DB cũ tự nâng cấp (`ADD COLUMN IF NOT EXISTS`, cột sinh tự động), không cần embed lại.
- **Tiêu cực / đánh đổi:**
  - Vùng α tốt khá hẹp (0.1–0.2) và chỉ được đo trên 39 câu trong phạm vi, nên có thể khớp quá sát golden set.
  - Không giúp được câu hỏi khác ngôn ngữ tài liệu (e19). Việc đó dành cho reranker.
  - Chi phí đếm df tăng theo số chunk chứa từ phổ biến.
- **Việc cần làm kèm theo:**
  - Đo lại α mỗi khi đổi embed model hoặc golden set lớn lên.
  - Reranker (M6a bước 2).
  - Nếu kho tài liệu lớn đến mức đếm df bị chậm: lưu sẵn bảng df, hoặc chuyển sang BM25 thật.

## Tham khảo
- [06 §6](../06-evaluation-and-testing.md#6-baseline): số đo dense, RRF và các giá trị α.
- Cormack et al., *Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods* (SIGIR 2009).
