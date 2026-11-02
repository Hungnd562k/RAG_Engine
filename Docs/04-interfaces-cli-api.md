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
| `rag ask "<q>" [--top-k N] [--show-context] [--json]` | Hỏi đáp có trích dẫn | M3 |
| `rag serve [--host] [--port]` | Chạy FastAPI | M4 |
| `rag eval <golden.yaml> [--top-k N] [--out FILE]` | Đánh giá retrieval | M5 |

Quy ước: `--json` in kết quả cùng schema với API để dễ so sánh/scripting. Exit code ≠ 0 khi lỗi hạ tầng.

### Ví dụ output `rag search`
```
 #  score  source › heading                               preview
 1  0.781  samples/hr-policy.md › Nghỉ phép › Năm          Nhân viên chính thức được 12 ngày...
 2  0.652  samples/hr-policy.md › Nghỉ phép › Ốm           Trường hợp nghỉ ốm cần giấy...
timings: embed=41ms search=6ms
```

## 2. REST API (FastAPI)

Base URL: `http://localhost:8000` · Swagger UI: `/docs` · OpenAPI: `/openapi.json`

### `GET /health`
```json
{ "status": "ok", "db": "ok", "ollama": "ok", "embed_model": "bge-m3", "llm_model": "qwen2.5:3b-instruct" }
```

### `GET /stats`
```json
{ "documents": 15, "chunks": 120, "embed_model": "bge-m3", "embed_dim": 1024 }
```

### `POST /ingest`
MVP nhận đường dẫn trên máy chạy server (local-only).
```json
// request
{ "path": "samples/", "prune": false }
// response
{ "added": 15, "updated": 0, "skipped": 0, "deleted": 0, "chunks": 120, "elapsed_ms": 25300 }
```

### `POST /search`
```json
// request
{ "query": "chính sách nghỉ phép", "top_k": 5, "source_prefix": null }
// response
{
  "results": [
    { "chunk_id": 42, "source": "samples/hr-policy.md", "heading_path": ["Nghỉ phép", "Năm"],
      "score": 0.781, "text": "Nhân viên chính thức được 12 ngày..." }
  ],
  "timings_ms": { "embed": 41, "search": 6, "total": 48 }
}
```

### `POST /ask`
```json
// request
{ "question": "Nhân viên được nghỉ phép bao nhiêu ngày?", "top_k": 5 }
// response
{
  "answer": "Nhân viên chính thức được 12 ngày phép/năm [1].",
  "sources": [ { "ref": 1, "chunk_id": 42, "source": "samples/hr-policy.md",
                 "heading_path": ["Nghỉ phép","Năm"], "score": 0.781 } ],
  "timings_ms": { "embed": 41, "search": 6, "generate": 3200, "total": 3260 }
}
```

### Lỗi
| HTTP | Khi nào | Body |
|------|---------|------|
| 400 | Input không hợp lệ (path không tồn tại, query rỗng) | `{"detail": "..."}` |
| 503 | DB hoặc Ollama không truy cập được | `{"detail": "ollama unreachable"}` |
| 409 | Model/dim lệch với dữ liệu đã lưu | `{"detail": "embed model mismatch, run reset"}` |
