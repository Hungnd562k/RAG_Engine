# 04 — Interfaces: CLI & REST API

CLI và API là hai lớp mỏng gọi cùng một engine (`build_engine(settings)`), nên kết quả phải giống nhau.

## 1. CLI (`rag`, Typer + rich)

| Lệnh | Mô tả | Milestone |
|------|-------|-----------|
| `rag health` | Kiểm tra DB, pgvector, Ollama, models; tạo schema nếu chưa có | M0 |
| `rag ingest <path> [--prune] [--dry-run]` | Nạp file/thư mục TXT/MD | M1 |
| `rag stats` | Số documents, chunks, model, dim | M1 |
| `rag reset [--yes]` | Xóa toàn bộ dữ liệu (hỏi xác nhận) | M1 |
| `rag search "<q>" [--top-k N] [--source PREFIX] [--min-score X] [--json]` | Tìm kiếm ngữ nghĩa | M2 |
| `rag ask "<q>" [--top-k N] [--source PREFIX] [--min-score X] [--show-context] [--json]` | Hỏi đáp có trích dẫn | M3 |
| `rag serve [--host 127.0.0.1] [--port 8000] [--reload]` | Chạy FastAPI | M4 |
| `rag mcp [--docs-root DIR]` | MCP server (stdio) cho agent: `search_docs`, `list_sources` ([§3](#3-mcp-server-rag-mcp)) | MCP pilot |
| `rag eval <golden.yaml>... [--top-k N] [--with-answer] [--all] [--out FILE]` | Đánh giá retrieval (hit@k, MRR, chặn OOS, latency); `--with-answer` chấm thêm câu trả lời và trích dẫn | M5 |

`rag ask` in cảnh báo đỏ khi câu trả lời có con số không xuất hiện trong context hoặc câu hỏi (dấu hiệu bịa, R11). Câu trả lời vẫn được trả về, chỉ gắn cờ.

Quy ước: `--json` in kết quả cùng schema với API để dễ so sánh/scripting. Exit code ≠ 0 khi lỗi hạ tầng.

### Ví dụ output `rag search`
```
 #  score  source › heading                               preview
 1  0.781  samples/hr-policy.md › Nghỉ phép › Năm          Nhân viên chính thức được 12 ngày...
 2  0.652  samples/hr-policy.md › Nghỉ phép › Ốm           Trường hợp nghỉ ốm cần giấy...
timings: embed=41ms search=6ms
```

## 2. REST API (FastAPI)

Chạy: `uv run rag serve [--host 127.0.0.1] [--port 8000] [--reload]` · Base URL: `http://localhost:8000` · Swagger UI: `/docs` · OpenAPI: `/openapi.json`

Code: `src/rag_engine/api/main.py` (`create_app()`), schema dùng chung với CLI `--json`: `src/rag_engine/schemas.py`.

- Engine khởi tạo **một lần** khi server start (lifespan), dùng connection pool Postgres (an toàn khi nhiều request song song). Nếu hạ tầng chưa lên lúc start, server vẫn chạy; `/health` báo lỗi và các endpoint khác tự thử khởi tạo lại.
- Mặc định chỉ nghe `127.0.0.1`: `/ingest` nhận đường dẫn bất kỳ trên máy chạy server, **không có auth** — không mở ra mạng ngoài (auth để M6+).

### `GET /health`
HTTP 200 nếu mọi thành phần bắt buộc OK (kể cả khi thiếu LLM → `degraded`), 503 nếu không.
```json
{
  "status": "ok",                       // ok | degraded | error
  "embed_model": "bge-m3",
  "llm_model": "qwen2.5:3b-instruct",
  "checks": [ { "name": "Postgres", "ok": true, "required": true, "detail": "localhost:5432/rag" }, ... ]
}
```

### `GET /stats`
```json
{ "documents": 7, "chunks": 33, "embed_model": "bge-m3", "embed_dim": 1024, "pgvector": "0.8.7" }
```

### `POST /ingest`
Đường dẫn tính trên máy chạy server, tương đối so với thư mục server được khởi động. Chỉ một ingest chạy tại một thời điểm.
```json
// request
{ "path": "samples", "prune": false, "dry_run": false }
// response
{ "added": 7, "updated": 0, "skipped": 0, "deleted": 0, "chunks": 33, "elapsed_ms": 6900.1, "dry_run": false,
  "details": [ { "source": "samples/hr-policy.md", "action": "added", "chunks": 6 }, ... ] }
```

### `POST /search`  — cùng schema với `rag search --json`
```json
// request (chỉ query là bắt buộc; top_k mặc định RAG_TOP_K, min_score mặc định RAG_MIN_SCORE)
{ "query": "chính sách nghỉ phép", "top_k": 5, "source_prefix": null, "min_score": null }
// response
{
  "results": [
    { "chunk_id": 6, "source": "samples/hr-policy.md", "heading_path": ["...", "Nghỉ phép", "Nghỉ phép năm"],
      "score": 0.7018, "lexical": 0.41, "rerank": 0.97, "text": "## Nghỉ phép ..." }
  ],
  "timings_ms": { "embed": 41.2, "search": 6.1, "rerank": 480.3, "total": 527.6 }
}
```
- `score` luôn là cosine. `lexical` (độ phủ từ khóa IDF, 0–1) chỉ có ở chế độ hybrid; `rerank` (cross-encoder, 0–1) và `timings_ms.rerank` chỉ có khi `RAG_RERANK_ENABLED=true`. Trường nào không áp dụng thì là `null` hoặc vắng mặt.
- Thứ tự `results` là thứ tự xếp hạng cuối cùng (rerank > điểm gộp hybrid > cosine), không nhất thiết giảm dần theo `score`.

### `POST /ask`  — cùng schema với `rag ask --json`
```json
// request
{ "question": "Mật khẩu phải dài tối thiểu bao nhiêu ký tự?", "top_k": 5, "source_prefix": null, "min_score": null }
// response
{
  "answer": "Mật khẩu tài khoản công ty phải dài tối thiểu 12 ký tự [1].",
  "answered": true,                // false khi bị chặn bởi min_score (không gọi LLM)
  "sources": [ { "ref": 1, "chunk_id": 12, "source": "samples/it-security.md",
                 "heading_path": ["...", "Mật khẩu"], "score": 0.737, "text": "..." } ],
  "cited": [1],                    // các [n] hợp lệ có trong answer
  "invalid_citations": [],         // các [n] không trỏ tới nguồn nào
  "unsupported_numbers": [],       // con số trong answer không có trong context/câu hỏi (M6a.3, R11)
  "timings_ms": { "embed": 61, "search": 5, "generate": 1984, "total": 2050 }
}
```

### Lỗi
| HTTP | Khi nào | Body |
|------|---------|------|
| 400 | Path không tồn tại, query không hợp lệ | `{"detail": "..."}` |
| 409 | Model/dim embedding lệch dữ liệu đã lưu; hoặc đang có ingest khác chạy | `{"detail": "..."}` |
| 422 | Sai schema request (vd. `query` rỗng, `top_k` ≤ 0 hoặc > 50) | chi tiết lỗi validate của FastAPI |
| 503 | Postgres/Ollama/reranker không truy cập được, model chưa pull | `{"detail": "..."}` |

## 3. MCP server (`rag mcp`)
MCP server qua stdio cho agent (Claude Code) tra cứu tài liệu đã ingest. Chỉ đọc ([ADR-0012](adr/0012-mcp-search-docs-for-agents.md)).

| Tool | Tham số | Trả về |
|------|---------|--------|
| `search_docs` | `query`, `top_k` (1–20, mặc định 5), `source_prefix` | Các chunk: `[n] <đường dẫn tuyệt đối> › heading`, `(score=… [rerank=…])`, cờ `[STALE]` (file đổi sau lần ingest) / `[MISSING]` (file đã xóa), nội dung chunk. Không có kết quả: thông báo kèm cosine cao nhất và gợi ý dùng grep |
| `list_sources` | `prefix` | Danh sách file đã ingest, kèm đường dẫn tuyệt đối |

- Server khởi động được khi Docker chưa chạy; tool báo lỗi rõ ràng (để agent chuyển sang grep) và thử kết nối lại ở lần gọi sau.
- Source tương đối được quy đổi theo `--docs-root` (mặc định thư mục hiện tại, tức thư mục chạy `rag ingest`).
- Retrieval dùng cùng cấu hình `.env` với CLI/API (`RAG_SEARCH_MODE`, `RAG_RERANK_ENABLED`, `RAG_MIN_SCORE`).

Đăng ký với Claude Code (phạm vi user, dùng được ở mọi dự án):
```bash
claude mcp add -s user rag-docs -- uv run --directory /path/to/RAG_Engine rag mcp
claude mcp list            # rag-docs: ✓ Connected
```
`--directory` vừa chọn môi trường uv vừa đặt thư mục làm việc, nên `.env` của RAG_Engine được đọc và source tương đối (`samples/...`) quy đổi đúng.
