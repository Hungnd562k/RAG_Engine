# 02 — Milestones

Mỗi milestone nhỏ, chạy được, có **Definition of Done (DoD)** kiểm chứng bằng lệnh cụ thể. Không chuyển milestone khi DoD chưa đạt.

| Milestone | Mục tiêu | Trạng thái |
|-----------|----------|-----------|
| M0 | Skeleton & Infra | ✅ Done (2026-11-02) |
| M1 | Ingestion vào pgvector | ✅ Done (2026-11-02) |
| M2 | Retrieval (`rag search`) | ✅ Done (2026-11-02) |
| M3 | Generation có trích dẫn (`rag ask`) | ⬜ Todo |
| M4 | REST API | ⬜ Todo |
| M5 | Evaluation & Tests | ⬜ Todo |
| M6+ | Mở rộng | 💤 Backlog |

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
- Prompt Builder theo [01-architecture.md §8](01-architecture.md#8-prompt-mặc-định-m3), cắt theo `RAG_CONTEXT_MAX_CHARS`.
- Orchestrator `ask()`: short-circuit khi không có chunk đạt `min_score`.
- `rag ask "<question>" [--top-k] [--show-context]` in: câu trả lời, danh sách nguồn `[n] source › heading (score)`, timing `embed/search/generate/total`.

**DoD**
```bash
uv run rag ask "Nhân viên được nghỉ phép bao nhiêu ngày?"
# Nhân viên chính thức được 12 ngày phép/năm [1]...
# Sources:
#  [1] samples/hr-policy.md › Nghỉ phép năm (0.78)
# timings: embed=45ms search=8ms generate=3200ms total=3260ms

uv run rag ask "Giá bitcoin hôm nay?"
# Không tìm thấy thông tin trong tài liệu.
```
- Mọi `[n]` trong câu trả lời trỏ tới nguồn có trong danh sách.

---

## M4 — REST API
**Mục tiêu:** engine dùng được qua HTTP.

**Scope**
- FastAPI app (`uv run rag serve` hoặc `uvicorn rag_engine.api.main:app`).
- `GET /health`, `POST /ingest`, `POST /search`, `POST /ask`, `GET /stats` — schema theo [04-interfaces-cli-api.md](04-interfaces-cli-api.md).
- Engine khởi tạo 1 lần (lifespan), dùng chung connection pool.

**DoD**
```bash
uv run rag serve &
curl -s localhost:8000/health
curl -s -X POST localhost:8000/ask -H 'content-type: application/json' \
     -d '{"question":"Nhân viên được nghỉ phép bao nhiêu ngày?"}' | jq
```
- Mở `http://localhost:8000/docs` thử được mọi endpoint.
- Kết quả `/search` trùng với `rag search` cho cùng query.

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

---

## M6+ — Backlog mở rộng (ưu tiên giảm dần)
1. **Hybrid search**: `tsvector` + GIN, RRF fusion → đo lại bằng `rag eval`.
2. **Reranker**: `bge-reranker-v2-m3` (cross-encoder) cho top-20 → top-5.
3. **Loaders**: PDF (`pymupdf`), DOCX (`python-docx`), HTML.
4. **Streaming** cho `/ask` (SSE) và CLI.
5. **Collections / namespace**, metadata filter tổng quát.
6. **Ingest bất đồng bộ** (job queue), xử lý thư mục lớn.
7. **Answer evaluation**: faithfulness/relevance bằng LLM-as-judge.
8. **Auth**, rate limit, đóng gói Docker cho app, CI.
