# 02 — Milestones

Mỗi milestone nhỏ, chạy được, có **Definition of Done (DoD)** kiểm chứng bằng lệnh cụ thể. Không chuyển milestone khi DoD chưa đạt.

| Milestone | Mục tiêu | Trạng thái |
|-----------|----------|-----------|
| M0 | Skeleton & Infra | ✅ Done (2026-11-02) |
| M1 | Ingestion vào pgvector | ✅ Done (2026-11-02) |
| M2 | Retrieval (`rag search`) | ✅ Done (2026-11-02) |
| M3 | Generation có trích dẫn (`rag ask`) | ✅ Done (2026-11-02) |
| M4 | REST API | ✅ Done (2026-11-02) |
| M5 | Evaluation & Tests | ✅ Done (2026-11-03) |
| M6+ | Mở rộng (chia cụm M6a–M6e) | 🟡 M6a xong, tiếp M6b |

---

## M0 — Skeleton & Infra
**Mục tiêu:** môi trường chạy được, app nói chuyện được với Postgres và Ollama.

**Scope**
- `docker-compose.yml`: `pgvector/pgvector:pg16`, `ollama/ollama` (volume lưu model, GPU optional).
- Pull model: `bge-m3` (M1), `qwen2.5:3b-instruct` (M3).
- `pyproject.toml` (uv, Python 3.12), cấu trúc `src/rag_engine/`, `ruff`, `pytest`.
- `config.py` (pydantic-settings), `.env.example`, logging.
- Lệnh `rag health`: kiểm tra kết nối DB, extension `vector`, Ollama reachable, 2 model đã có.

**DoD** — ✅ kết quả thực tế:
```bash
docker compose up -d
uv run rag health
#  Postgres                 OK    localhost:5432/rag
#  pgvector                 OK    v0.8.7
#  Schema/meta              OK    bge-m3, dim=1024
#  Ollama                   OK    http://localhost:11434
#  Embed bge-m3             OK    dim=1024
#  LLM qwen2.5:3b-instruct  WARN  chưa pull (cần từ M3)
```

---

## M1 — Ingestion
**Mục tiêu:** nạp tài liệu TXT/MD thành chunks + embeddings trong pgvector.

**Scope**
- Loader TXT/MD, `content_hash` sha256, `source` = đường dẫn tương đối.
- Chunker: tách theo heading MD → recursive split (đoạn → câu → từ), `chunk_size`/`overlap` từ config, lưu `heading_path`, `char_start/end`.
- Embedder Ollama (batch, retry đơn giản, kiểm tra dim).
- `schema.sql` + `VectorStore.replace_document()` trong 1 transaction.
- Ingestion pipeline idempotent; lệnh `rag ingest <path>`, `rag stats`, `rag reset`.
- Thư mục `samples/` với 7 file MD/TXT (5 tiếng Việt + 2 tiếng Anh) có nội dung tra cứu được.

**DoD** — ✅ kết quả thực tế (CPU):
```bash
uv run rag ingest samples/
# added=7 updated=0 skipped=0 deleted=0 chunks=33 elapsed=6.9s
uv run rag stats
# documents=7 chunks=33 embed_model=bge-m3 embed_dim=1024 pgvector=0.8.7
uv run rag ingest samples/
# added=0 updated=0 skipped=7 deleted=0 chunks=0 elapsed=0.0s   ← idempotent
# (file tạm samples/zz-tmp.md) thêm → added=1; sửa → updated=1; xóa + --prune → deleted=1
```
Kiểm tra thủ công bằng SQL: `SELECT source, count(*) FROM chunks JOIN documents ... GROUP BY source;`

---

## M2 — Retrieval
**Mục tiêu:** tìm kiếm ngữ nghĩa, thấy ngay chunk nào được lấy và vì sao.

**Scope**
- Index HNSW `vector_cosine_ops`.
- Retriever: top-k, `min_score`, filter `--source` (prefix).
- `rag search "<query>" [--top-k 5] [--source samples/x]` in bảng: rank, score, source › heading, preview 200 ký tự.
- Tối thiểu 5 câu hỏi mẫu viết sẵn trong `eval/golden.yaml` (dùng lại ở M5).

**DoD** — ✅ kết quả thực tế:
```bash
uv run rag search "chính sách nghỉ phép năm" -k 3
# 1 0.702 samples/hr-policy.md › … › Nghỉ phép › Nghỉ phép năm   "Nhân viên chính thức được hưởng 12 ngày…"
# 2 0.570 samples/hr-policy.md › … › Nghỉ phép › Nghỉ ốm
# 3 0.538 samples/hr-policy.md › … › Nghỉ phép › Nghỉ thai sản
# timings: embed=41ms search=2ms total=43ms
uv run rag search "Giá bitcoin hôm nay là bao nhiêu?"
# Không có chunk nào đạt ngưỡng min_score.
```
- 6/6 câu golden có nguồn đúng ở **rank 1** (tự động hóa trong `tests/integration/test_live_stack.py`, kiểm tra top-3).
- Cross-lingual: "Làm việc từ nước ngoài được tối đa bao nhiêu ngày?" → `remote-work-policy.md › Working abroad` (0.746).
- Ngưỡng `min_score` hiệu chỉnh 0.35 → 0.50 dựa trên đo đạc ([Q6](07-risks-and-open-questions.md#q6--ngưỡng-min_score)).

---

## M3 — Generation có trích dẫn
**Mục tiêu:** hỏi – đáp end-to-end.

**Scope**
- `LLMProvider` + `OllamaLLM` (`/api/chat`, `stream=false`, temperature thấp).
- Prompt Builder theo [01-architecture.md §8](01-architecture.md#8-prompt-mặc-định-m3), cắt theo `RAG_CONTEXT_MAX_CHARS`; kiểm tra trích dẫn `[n]` hậu kỳ (hợp lệ / không hợp lệ).
- Orchestrator `ask()`: short-circuit khi không có chunk đạt `min_score`.
- `rag ask "<question>" [--top-k] [--source] [--min-score] [--show-context] [--json]` in: câu trả lời, danh sách nguồn `[n] source › heading (score)`, timing `embed/search/generate/total`.

**DoD** — ✅ kết quả thực tế (qwen2.5:3b-instruct trên GPU, bge-m3 trên CPU, model đã nạp):
```bash
uv run rag ask "Is SMS allowed for 2FA?"
# SMS không được chấp nhận làm yếu tố thứ hai trong xác thực hai yếu tố (2FA) [1].
# Sources:
#  ✓ [1] samples/it-security.md › Quy định an toàn thông tin › Xác thực hai yếu tố (…)
# timings: embed=52ms search=3ms generate=1531ms total=1585ms

uv run rag ask "Giá bitcoin hôm nay?"
# Không tìm thấy thông tin trong tài liệu.        ← chặn bởi min_score, không gọi LLM
# timings: embed=50ms search=2ms total=53ms
```
- `tests/integration/test_live_stack.py::test_ask_golden_answers_and_citations`: mọi câu golden có `answer_contains` đúng, ≥ 1 trích dẫn, không trích dẫn sai; câu ngoài phạm vi bị chặn.
- CLI đánh dấu ✓ nguồn được trích dẫn, cảnh báo khi thiếu trích dẫn hoặc có `[n]` không hợp lệ.
- Generate 0.6–6 s/câu khi model đã nạp; lần đầu sau khi khởi động thêm vài giây để nạp model.

**Phát hiện khi triển khai (đã xử lý)**
1. *Tranh chấp VRAM:* `bge-m3` + `qwen2.5:3b` không cùng vừa 4 GB → Ollama đẩy model ra/vào mỗi câu hỏi, embed 2–6 s. Giải pháp: `RAG_EMBED_NUM_GPU=0` (embed chạy CPU ~50 ms/query), `RAG_LLM_NUM_CTX=4096`. LLM vẫn chạy 76% GPU / 24% CPU.
2. *Model 3B bỏ trích dẫn / từ chối quá mức:* prompt ban đầu chỉ 1/3 câu có `[n]`; quy tắc "không đủ thông tin thì nói không tìm thấy" khiến model từ chối cả câu trả lời được (câu mơ hồ, câu hỏi EN trên tài liệu VI). A/B 4 biến thể prompt trên 12 câu: bỏ quy tắc từ chối (đã có `min_score` chặn trước) + ví dụ mẫu + nhắc lại ở cuối → **12/12** ở top_k 3 và 5 (trước: 9/10).
3. *Giới hạn còn lại của 3B:* câu trả lời đôi khi dài dòng/rào đón thừa; có lúc trả lời bằng tiếng Việt cho câu hỏi tiếng Anh. Thử `RAG_LLM_MODEL=qwen2.5:7b-instruct` nếu cần chất lượng cao hơn (chậm hơn trên GPU 4 GB).

---

## M4 — REST API
**Mục tiêu:** engine dùng được qua HTTP.

**Scope**
- FastAPI app (`uv run rag serve` hoặc `uvicorn rag_engine.api.main:app`).
- `GET /health`, `POST /ingest`, `POST /search`, `POST /ask`, `GET /stats` — schema theo [04-interfaces-cli-api.md](04-interfaces-cli-api.md).
- Engine khởi tạo 1 lần (lifespan), dùng chung connection pool.

**DoD** — ✅ kết quả thực tế:
```bash
uv run rag serve                                   # http://127.0.0.1:8000/docs
curl -s localhost:8000/health                      # 200, status=ok, 6/6 checks OK
curl -s -X POST localhost:8000/ask -H 'content-type: application/json' \
     -d '{"question":"Mật khẩu phải dài tối thiểu bao nhiêu ký tự?","top_k":3}'
# "Mật khẩu tài khoản công ty phải dài tối thiểu 12 ký tự [1]."  cited=[1]
# timings: embed=61 search=5 generate=1984 total=2050 (ms)
```
- `/docs` (Swagger UI) trả 200, thử được mọi endpoint.
- `/search` và `rag search --json` cho cùng query: **cùng schema, cùng chunk_id/source/score** (kiểm tra tự động trong `tests/unit/test_api.py` và `tests/integration/test_live_stack.py::test_api_against_live_engine`).
- 40 request `/search` từ 16 luồng song song: 100% HTTP 200 trong 2.2 s, kết quả ổn định.
- Lỗi: path không tồn tại → 400, query rỗng → 422, câu ngoài phạm vi → `answered=false`.

**Thay đổi kỹ thuật kèm theo**
- `PgVectorStore` chuyển từ 1 connection sang **connection pool** (`psycopg_pool`, 1–4 connections): mỗi thao tác mượn connection riêng, transaction của ingest không lẫn với search song song.
- `src/rag_engine/schemas.py`: Pydantic models dùng chung cho API và CLI `--json`.
- `/ingest` có khóa: chỉ 1 ingest chạy cùng lúc (request thứ hai nhận 409).

---

## M5 — Evaluation & Tests
**Mục tiêu:** đo được chất lượng, có lưới an toàn khi thay đổi.

**Scope**
- Unit tests: chunker (biên, overlap, heading_path), loader, prompt builder, parse config.
- Integration tests (`@pytest.mark.integration`): store (DB thật qua compose), embedder/LLM (Ollama thật).
- `eval/golden.yaml` ≥ 20 câu (gồm câu ngoài phạm vi).
- `rag eval eval/golden.yaml` in hit@1, hit@3, hit@5, MRR, latency p50/p95; xuất `eval/report-<date>.json`.
- Tinh chỉnh `chunk_size`, `overlap`, `min_score` dựa trên số đo; ghi baseline vào [06-evaluation-and-testing.md](06-evaluation-and-testing.md).

**DoD**
```bash
uv run pytest -m "not integration"      # xanh, < 5s
uv run pytest -m integration            # xanh khi compose đang chạy
uv run rag eval eval/golden.yaml
# hit@1=0.70 hit@3=0.85 hit@5=0.90 MRR=0.78 search_p50=12ms
```
- hit@5 ≥ 0.8 (theo [00-vision-and-scope.md](00-vision-and-scope.md)).

**Kết quả (2026-11-03)**
- `uv run pytest -m "not integration"`: 43 passed trong ~0.5 s. `uv run pytest -m integration`: 4 passed trong ~34 s. Integration test chấm theo ngưỡng (hit@5, chặn OOS) thay vì đòi đúng từng câu.
- `rag eval eval/golden.yaml` (19 câu trong phạm vi + 6 OOS): hit@1 = hit@3 = hit@5 = MRR = **1.00**, OOS block 6/6, search p50 4 ms / p95 6 ms. Bộ đối chứng E-Wallet riêng tư: hit@5 0.90. → **hit@5 ≥ 0.8 đạt.**
- `--with-answer`: answer_ok 0.947, citation_ok 1.00 (samples); 0.95 / 0.90 (E-Wallet).
- Tinh chỉnh: giữ chunk 512/64 và `min_score` 0.50. Đã thử hai biến thể prompt, cả hai đều kém hơn nên giữ prompt M3. Chi tiết ở [06 §6](06-evaluation-and-testing.md#6-baseline).
- Phát hiện chuyển sang M6: khi retrieval trượt, 3B bịa con số (R11 trong [07](07-risks-and-open-questions.md)). Đây là lý do nâng ưu tiên cho hybrid search/reranker và hậu kiểm con số trong câu trả lời.

---

## M6+ — Mở rộng, chia thành cụm

Mỗi cụm là một nhánh và một PR riêng. Mọi thay đổi về retrieval phải đo bằng `rag eval` và thêm một dòng vào [06 §6](06-evaluation-and-testing.md#6-baseline).

| Cụm | Nội dung | Trạng thái |
|-----|----------|-----------|
| **M6a**: Chất lượng retrieval | 1. Hybrid search ✅<br>2. Reranker `bge-reranker-v2-m3` cho top-20 → top-5 ✅<br>3. Hậu kiểm con số trong câu trả lời phải có trong context (R11) ✅ | ✅ Done |
| **M6b**: Đánh giá answer | LLM-as-judge (faithfulness/relevance) | ⬜ |
| **M6c**: Nạp dữ liệu rộng hơn | Công cụ migration schema; loader PDF (`pymupdf`), DOCX, HTML; collections và lọc metadata; ingest bất đồng bộ | ⬜ |
| **M6d**: Trải nghiệm hỏi đáp | Streaming `/ask` (SSE) và CLI; hội thoại nhiều lượt + viết lại câu hỏi (Q3) | ⬜ |
| **M6e**: Vận hành | Auth, ACL (Q4), rate limit, image Docker cho app, CI | ⬜ |

Cho agent (Claude) gọi `engine.search()` qua MCP `search_docs` **sau khi** retrieval ổn định và chính xác (sau M6a): đang chạy pilot, xem [MCP pilot](#mcp-pilot--search_docs-cho-agent-2026-11-09).

### M6a.1 — Hybrid search (2026-11-03)
- Cột `content_tsv` (`to_tsvector('simple', content)`, GIN). Gộp `cosine + 0.15 × độ phủ IDF`, không dùng RRF ([ADR-0010](adr/0010-hybrid-search-score-fusion.md)).
- `RAG_SEARCH_MODE=dense|hybrid` (mặc định `hybrid`), `RAG_HYBRID_ALPHA`, `RAG_HYBRID_CANDIDATES`.
- **DoD:** `RAG_SEARCH_MODE=hybrid uv run rag eval eval/golden.yaml` không kém dense. Bộ E-Wallet tốt hơn: hit@1 tăng từ 0.90 lên 0.95, e14 từ không có trong top 5 lên hạng 1. Tỉ lệ chặn OOS không đổi.

### M6a.2 — Reranker (2026-11-03)
- Service `reranker` (TEI, `bge-reranker-v2-m3`). GPU qua `docker-compose.gpu.yml` (image `86-1.7`). `TEIReranker` tự chia batch 32 ([ADR-0011](adr/0011-reranker-tei-service.md)).
- `RAG_RERANK_ENABLED` (mặc định `false`), `RAG_RERANK_CANDIDATES=20`. `RAG_SEARCH_MODE=auto`: dense khi bật rerank, hybrid khi tắt. `rag health` kiểm tra reranker. `rag eval` báo `rerank_p50/p95`.
- **DoD:** `RAG_RERANK_ENABLED=true uv run rag eval eval/golden.yaml` không kém. Bộ E-Wallet: hit@5 tăng từ 0.95 lên 1.00 (e19 vào top 5), answer_ok và citation_ok đều đạt 1.00. Rerank p50 khoảng 0.5–0.7 s trên GPU.

### M6a.3 — Hậu kiểm con số (2026-11-09)
- `generation/grounding.py`: `unsupported_numbers(answer, sources, question)` trả về các con số trong câu trả lời không có trong context (gồm source/heading) hoặc câu hỏi, so khớp sau chuẩn hóa dấu phân cách. Chỉ gắn cờ, không chặn câu trả lời.
- `Answer.unsupported_numbers`, trường `unsupported_numbers` trong `/ask` và `rag ask --json`; `rag ask` in cảnh báo đỏ. `rag eval --with-answer` báo `numbers_grounded` và liệt kê câu bị gắn cờ.
- **DoD:** không có cảnh báo nhầm trên `samples` và E-Wallet (numbers_grounded 1.00 ở mode hybrid). Mode dense tái hiện lỗi bịa giờ ở e14 và bị gắn cờ (`17:00`). Chi tiết ở [06 §6.2](06-evaluation-and-testing.md#62-answer---with-answer-qwen253b-instruct-gpu).

### MCP pilot — `search_docs` cho agent (2026-11-09)
- `rag mcp`: MCP server stdio, chỉ đọc, tool `search_docs` và `list_sources` ([ADR-0012](adr/0012-mcp-search-docs-for-agents.md), [04 §3](04-interfaces-cli-api.md#3-mcp-server-rag-mcp)). Kết quả có đường dẫn tuyệt đối, cờ `STALE`/`MISSING`, thông báo gợi ý grep khi không có kết quả; lỗi hạ tầng trả `ToolError` rõ ràng.
- Golden kiểu agent: `eval/golden-agent.yaml` (+ bộ E-Wallet riêng tư).
- Sửa kèm: `rag search/ask/eval` và API bắt `RerankError` (trước đây in traceback).
- **DoD:** hit@5 = 1.00 trên cả hai bộ câu kiểu agent ở cấu hình mặc định; smoke test qua stdio với client MCP thật; đăng ký vào Claude Code và dùng thử trên dự án E-Wallet.
