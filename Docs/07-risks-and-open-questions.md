# 07 — Rủi ro & Câu hỏi mở

## 1. Rủi ro

| # | Rủi ro | Ảnh hưởng | Khả năng | Giảm thiểu |
|---|--------|-----------|----------|------------|
| R1 | GPU chỉ 4 GB VRAM (RTX 3050 Laptop); `qwen2.5:7b` (~4.7 GB) không vừa VRAM (NVIDIA runtime đã cài, Ollama chạy GPU) | Generation chậm, trải nghiệm "thấy ngay" kém | Cao | Mặc định đổi sang `qwen2.5:3b-instruct` (~1.9 GB, vừa VRAM cùng `bge-m3` ~1.2 GB); bật GPU bằng `docker-compose.gpu.yml` sau khi cài nvidia-container-toolkit; `rag search` vẫn nhanh để kiểm chứng retrieval độc lập |
| R2 | Ollama chỉ trả dense vector của BGE-M3, mất sparse/ColBERT | Recall với từ khóa hiếm/mã số kém | ~~Trung bình~~ Thấp | ✅ M6a: hybrid search gộp cosine + độ phủ IDF ([ADR-0010](adr/0010-hybrid-search-score-fusion.md)), e14 `recon.daily` lên hạng 1. Câu khác ngôn ngữ tài liệu (e19): ✅ reranker đưa vào top 5 (M6a.2) |
| R3 | Dimension lock-in (vector(1024)) | Đổi model phải re-embed toàn bộ | Thấp (MVP) | `engine_meta` phát hiện lệch; `rag reset` + ingest lại |
| R4 | Chunk size tính theo ký tự ước lượng, sai khác token thật với tiếng Việt | Chunk quá dài/ngắn | ~~Trung bình~~ Thấp | Đã đo ở M5: chunk 256–768 cho hit@1 như nhau (0.949), giữ 512/64 ([06 §6.3](06-evaluation-and-testing.md#63-tinh-chỉnh-tham-số-m5)). Chỉ cần tokenizer thật khi tài liệu dài, ít heading |
| R5 | LLM bịa hoặc trích dẫn sai `[n]` | Mất tin cậy | Trung bình | ✅ Đã triển khai ở M3: `min_score` short-circuit (không gọi LLM), prompt grounding + few-shot, temperature 0.1, `extract_citations` tách `[n]` hợp lệ/không hợp lệ và CLI cảnh báo. Còn lại: 3B đôi khi dài dòng/sai ngôn ngữ trả lời. M5 đo: citation_ok 1.00 (samples) / 0.90 (E-Wallet) |
| R11 | Retrieval trượt nhưng chunk vẫn vượt `min_score` → 3B **bịa con số** thay vì nói không biết (E-Wallet e14: "recon.daily chạy lúc 8:00"). Ví dụ few-shot có thể bị chép (số "8:00–17:00") | Câu trả lời sai mà vẫn có trích dẫn, rất khó phát hiện | Trung bình | Đã thử ví dụ trung tính ở M5: vẫn bịa và phát sinh lỗi `[n.1]` → giữ prompt M3. M6a: hybrid ✅ (e14 tìm đúng và trả lời đúng "01:00"), reranker để giảm trượt tiếp; ✅ M6a.3 hậu kiểm con số: số không có trong context/câu hỏi bị gắn cờ (`unsupported_numbers`, `numbers_grounded`); tái hiện e14 ở mode dense thì bị bắt. Còn lại: số có trong context nhưng chọn sai (để M6b LLM-as-judge) |
| R9 | Embed + LLM (+ reranker từ M6a) tranh 4 GB VRAM → Ollama nạp/đẩy model liên tục | Mỗi câu hỏi chậm thêm 2–6 s | ~~Cao~~ Đã xử lý | `RAG_EMBED_NUM_GPU=0` (embed trên CPU), `RAG_LLM_NUM_CTX=4096`. M6a: reranker GPU chiếm ~2.4 GB → `qwen2.5:3b` bị đẩy 74% sang CPU, generate chậm ~2.5 lần; `RAG_RERANK_ENABLED` mặc định tắt, `search` (đường agent dùng) không bị ảnh hưởng ([ADR-0011](adr/0011-reranker-tei-service.md)) |
| R10 | Prompt vượt `num_ctx` → Ollama âm thầm cắt đầu prompt (mất system prompt) | Mất grounding, mất trích dẫn | Trung bình | `RAG_CONTEXT_MAX_CHARS=8000` ≲ `num_ctx × 2`; M5 cân nhắc đếm token thật trước khi gửi |
| R6 | Ollama thay đổi API/hành vi `/api/embed` (normalize, truncate) | Kết quả lệch | Thấp | Pin image version khi ổn định; integration test kiểm tra dim/cosine |
| R7 | Python 3.14 hệ thống thiếu wheel cho vài lib | Cài đặt lỗi | ~~Trung bình~~ Đã xử lý | Pin 3.12 qua uv (`.python-version`), `uv sync` chạy sạch |
| R8 | Golden set nhỏ, chủ quan | Metrics không đại diện | ~~Cao~~ Trung bình | M5: `samples` 25 câu + bộ E-Wallet riêng tư 24 câu (tài liệu thật của dự án khác, `eval/local/`, gitignored). Vẫn coi là smoke metric; integration test chấm theo ngưỡng thay vì đòi đúng 100% |

## 2. Câu hỏi mở & trả lời

| # | Câu hỏi | Cần trả lời trước | Trạng thái |
|---|---------|-------------------|-----------|
| Q1 | Dữ liệu thật đầu tiên là gì (domain, ngôn ngữ, khối lượng)? Ảnh hưởng chunk size & golden set | M5 | ✅ Đã chốt (M5) |
| Q2 | Engine sẽ được dùng như **thư viện** import hay **service** HTTP là chính? | M4 | ✅ Đã chốt |
| Q3 | Có cần lưu lịch sử hội thoại (multi-turn, query rewriting)? | M6 | ✅ Đã chốt |
| Q4 | Yêu cầu phân quyền theo tài liệu/người dùng? | Trước khi dùng dữ liệu nhạy cảm | ✅ Đã chốt |
| Q5 | Có GPU không? Quyết định model LLM mặc định | M0 | ✅ Đã chốt |
| Q6 | Ngưỡng `min_score` 0.35 có hợp với BGE-M3 trên dữ liệu thật? | M5 | ✅ Đã chốt 0.50 (M5) |

### Q1 — Dữ liệu đầu tiên
**Tạm chốt:** dùng bộ `samples/` — 7 tài liệu nội bộ giả lập của "Công ty Sao Biển" (chính sách nhân sự, an toàn thông tin, công tác phí, hội nhập, làm việc từ xa, FAQ sản phẩm, ghi chú văn phòng), trộn **tiếng Việt (5) + tiếng Anh (2)**, MD + TXT, ~60 chunks. Đặc trưng giống tài liệu nội bộ doanh nghiệp: nhiều con số, định mức, quy trình — phù hợp để kiểm chứng trích dẫn chính xác.
**Cập nhật M5 (2026-11-03):** đã thêm bộ đối chứng là tài liệu thiết kế thật của dự án E-Wallet (FSD/URD/ERD/business rules/quy trình review, tiếng Việt, MD). Golden set 24 câu đặt ở `eval/local/` và không commit. Kết quả hit@1 0.90 (hai câu trượt). Chunk size đã đo lại và giữ nguyên 512/64. Dữ liệu chính của engine vẫn là tài liệu thiết kế của chính repo này (Docs/), đúng theo mục đích tra cứu tài liệu cho agent.

### Q2 — Thư viện hay service
**Chốt: library-first, service là cách triển khai chính.** Lõi là package `rag_engine` với factory `build_engine(settings)`, import trực tiếp được (đã dùng trong integration test). CLI và FastAPI (M4) chỉ là lớp mỏng gọi `build_engine`. Vì mục tiêu repo là "reusable across multiple applications", các ứng dụng khác sẽ tích hợp qua HTTP API (không phụ thuộc Python, tách vòng đời model/DB khỏi app). → Ràng buộc: không đặt logic nghiệp vụ trong `cli.py`/`api/`.

### Q3 — Hội thoại nhiều lượt
**Chốt: không cho MVP (M0–M5), single-turn.** Đưa vào M6 dạng tùy chọn: API `/ask` nhận thêm `history: [{role, content}]`, bước *query rewriting* dùng LLM viết lại câu hỏi phụ thuộc ngữ cảnh ("thế còn nghỉ ốm?") thành câu độc lập trước khi retrieve. Engine vẫn stateless — client tự giữ lịch sử.

### Q4 — Phân quyền
**Chốt: không cho MVP** — local, một người dùng, dữ liệu mẫu không nhạy cảm. Trước khi nạp dữ liệu thật có phân quyền: thêm `documents.metadata.acl` (danh sách group được xem) + filter SQL trong `VectorStore.search` theo group của người gọi, và auth ở API (M6). Lọc phải nằm **trong truy vấn DB** (trước top-k), không lọc sau khi đã đưa vào prompt.

### Q5 — GPU & model mặc định
**Phần cứng (đo 2026-11-02):** NVIDIA GeForce RTX 3050 Laptop GPU **4 GB VRAM** (driver 595.99), RAM 22 GB, 12 luồng CPU. Ban đầu Docker chưa có NVIDIA runtime → Ollama chạy CPU. **Đã cài nvidia-container-toolkit (2026-11-02)**: Ollama nhận CUDA (compute 8.6), `bge-m3` chạy 100% GPU; ingest samples 6.9 s → 0.6 s, embed query ~45 ms → ~15 ms.

**Chốt:**
- LLM mặc định: **`qwen2.5:3b-instruct`** (~1.9 GB Q4). `bge-m3` (~1.2 GB) + 3B vừa đủ 4 GB VRAM khi bật GPU; trên CPU vẫn chấp nhận được (ước tính 5–10 token/s).
- `qwen2.5:7b-instruct` (~4.7 GB) **không vừa** 4 GB VRAM → bị offload một phần sang CPU, chậm; chỉ dùng khi cần chất lượng cao hơn: `RAG_LLM_MODEL=qwen2.5:7b-instruct`.
- Bật GPU: cài `nvidia-container-toolkit` (cần sudo), rồi `docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d` — xem [05-local-setup.md](05-local-setup.md#gpu-nvidia).
- Việc đổi model mặc định là đổi cấu hình, nằm trong phạm vi [ADR-0007](adr/0007-llm-ollama-qwen2-5.md) (không cần ADR mới).

### Q6 — Ngưỡng `min_score`
**Chốt tạm: `RAG_MIN_SCORE = 0.50`** (thay cho 0.35). Đo trên `samples/` (33 chunks, bge-m3, ngày 2026-11-02), top-1 cosine:

| Nhóm | Số câu | Top-1 score min | Top-1 score max |
|------|--------|-----------------|-----------------|
| Trong phạm vi (VI, EN, cross-lingual, diễn đạt khác tài liệu) | 16 | **0.595** | 0.746 |
| Ngoài phạm vi (bitcoin, thời tiết, nấu ăn, bóng đá, vật lý…) | 8 | 0.328 | **0.460** |

- 0.35 **không chặn được** 6/8 câu ngoài phạm vi → sẽ đưa context rác vào LLM.
- 0.50 nằm giữa khoảng trống 0.460–0.595: chặn 8/8 câu ngoài phạm vi, giữ 16/16 câu trong phạm vi.
- Câu sát ngưỡng nhất: "Cách thay dầu xe máy" = 0.460 (trùng từ "xe máy" với ghi chú bãi gửi xe) → ngưỡng dense thuần dễ bị nhiễu bởi trùng từ vựng.

**Đo lại ở M5 (2026-11-03) bằng `rag eval`:** trên `samples/`, câu trong phạm vi thấp nhất 0.583 và OOS cao nhất 0.479, chặn 6/6. Trên E-Wallet, trong phạm vi thấp nhất 0.551 và OOS cao nhất 0.516 ("cài Kubernetes trên Ubuntu" lọt, nhưng LLM vẫn trả lời không tìm thấy). Giữ 0.50: nâng ngưỡng sẽ tiến sát câu trong phạm vi. Ngưỡng dense thuần không tách sạch được khi có trùng từ vựng; hướng xử lý là hybrid/reranker ở M6. Ở M3, ngoài ngưỡng còn có lớp bảo vệ thứ hai là prompt "không đủ thông tin thì nói không biết".

> Phát hiện phụ khi đo: file TXT không heading ngắn hơn `chunk_size` bị gộp thành 1 chunk nhiều chủ đề → câu hỏi về gửi ô tô rơi xuống rank 4 (score 0.448). Đã sửa chunker: TXT tách theo đoạn văn (xem addendum [ADR-0006](adr/0006-chunking-strategy.md)) → rank 1 (0.690).

