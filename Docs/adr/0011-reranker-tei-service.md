# ADR-0011: Reranker `bge-reranker-v2-m3` chạy trong container TEI riêng (GPU), mặc định tắt

- **Trạng thái:** Accepted
- **Ngày:** 2026-11-03

## Bối cảnh
Sau hybrid search ([ADR-0010](0010-hybrid-search-score-fusion.md)), bộ E-Wallet còn câu e19 trượt top 5. Câu này hỏi tiếng Anh vào tài liệu tiếng Việt, nên full-text không giúp được. Cross-encoder đọc câu hỏi và chunk cùng lúc nên chấm điểm liên quan chính xác hơn so sánh cosine giữa hai vector.

Ollama (đang phục vụ embed và LLM, [ADR-0004](0004-embedding-bge-m3-via-ollama.md), [ADR-0007](0007-llm-ollama-qwen2-5.md)) không có API rerank, nên reranker phải chạy ở chỗ khác. Phần cứng chỉ có 4 GB VRAM, và LLM 3B đã chiếm khoảng 2.8 GB (R1, R9).

## Quyết định
- **Service `reranker`** trong compose: HuggingFace Text Embeddings Inference (TEI), model `BAAI/bge-reranker-v2-m3`. App gọi `POST /rerank` qua `TEIReranker`, tự chia batch 32 text mỗi request vì vượt giới hạn mặc định của TEI sẽ nhận HTTP 413.
- **Chạy GPU** bằng image `86-1.7` trong `docker-compose.gpu.yml`:
  - Bản CPU mất khoảng 12 s cho 20 ứng viên, quá chậm.
  - `86-1.8` lỗi nạp XLM-RoBERTa (`cannot find tensor embeddings.word_embeddings.weight`).
  - `docker-compose.yml` vẫn khai báo bản CPU với `--max-batch-tokens 8192`; giá trị mặc định 16384 làm warm-up bị hết RAM (exit 137).
- **Luồng xử lý:** ứng viên (`RAG_RERANK_CANDIDATES`=20) → cross-encoder → top-k theo điểm rerank. Bước chặn câu ngoài phạm vi vẫn theo cosine (`min_score`). `score` trả ra vẫn là cosine, điểm rerank nằm ở trường `rerank`.
- **`RAG_SEARCH_MODE=auto`** (mặc định): dùng ứng viên **dense** khi bật rerank, **hybrid** khi tắt. Lý do: khi đã rerank, hybrid không cải thiện thêm mà còn chiếm chỗ ứng viên (e19 rơi ra ngoài 20 ứng viên), còn rerank toàn bộ 40 ứng viên của hybrid thì chậm gấp 2–3 lần mà chất lượng như nhau.
- **`RAG_RERANK_ENABLED` mặc định `false`**: engine chạy được mà không cần thêm service và không tranh VRAM. Bật khi có GPU.
- **Lỗi reranker → `RerankError`**, không âm thầm bỏ qua rerank, để số đo và hành vi không bị lệch mà không ai biết.

## Các phương án đã cân nhắc
| Phương án | Ưu | Nhược |
|-----------|----|-------|
| Container TEI riêng, GPU (chọn) | App không phải cài torch. Cùng mô hình infra với [ADR-0008](0008-local-infra-docker-compose.md). 20 ứng viên mất ~0.5–0.7 s | Chiếm ~2.4 GB VRAM: LLM 3B bị đẩy 74% sang CPU, generate chậm khoảng 2.5 lần. Warm-up lần đầu tải ~2.2 GB model |
| Container TEI, CPU | Không tranh VRAM | ~0.6 s cho mỗi chunk, 20 ứng viên mất 12–14 s |
| Chạy trong process của app (`sentence-transformers` / `FlagEmbedding`) | Không cần thêm service | Kéo theo torch (~2 GB), vẫn tranh VRAM, app nặng |
| Dùng LLM để rerank | Không cần model mới | Chậm, model 3B chấm điểm kém ổn định |

## Hệ quả
- **Tích cực** (E-Wallet, dense + rerank 20, so với hybrid không rerank):
  - hit@5 tăng từ 0.95 lên **1.00**, MRR từ 0.950 lên **0.967**. e19 lần đầu vào top 5 (hạng 3).
  - answer_ok tăng từ 0.95 lên **1.00**, citation_ok từ 0.90 lên **1.00**.
  - Bộ `samples` giữ nguyên 1.00.
- **Tiêu cực / đánh đổi:**
  - `search` chậm thêm khoảng 0.5–0.7 s (rerank p50).
  - `ask` chậm khoảng 2.5 lần vì LLM bị đẩy sang CPU.
  - Thêm một service và khoảng 2.2 GB model trong volume `hf`.
  - Image TEI bị pin ở 1.7 cho tới khi lỗi của 1.8 được sửa.
- **Việc cần làm kèm theo:**
  - Cân nhắc dùng điểm rerank để chặn câu ngoài phạm vi (eo2 lọt ở cosine 0.516).
  - Khi có GPU lớn hơn hoặc dùng LLM ngoài (Claude), tranh chấp VRAM không còn.
  - Hậu kiểm con số (M6a.3).

## Tham khảo
- [06 §6](../06-evaluation-and-testing.md#6-baseline): bảng so sánh dense / hybrid / rerank.
- BAAI `bge-reranker-v2-m3`, HuggingFace `text-embeddings-inference`.
