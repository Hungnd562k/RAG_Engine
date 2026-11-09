# Architecture Decision Records

Mỗi quyết định kiến trúc quan trọng được ghi thành một file `NNNN-tieu-de.md` theo [template.md](template.md).
Trạng thái: `Proposed` → `Accepted` → (`Deprecated` | `Superseded by NNNN`). Không sửa nội dung ADR đã `Accepted`; muốn đổi thì tạo ADR mới.

| # | Tiêu đề | Trạng thái | Ngày |
|---|---------|-----------|------|
| [0001](0001-record-architecture-decisions.md) | Ghi lại quyết định kiến trúc bằng ADR | Accepted | 2026-11-02 |
| [0002](0002-python-stack-and-tooling.md) | Python 3.12 + uv và bộ thư viện nền | Accepted | 2026-11-02 |
| [0003](0003-no-rag-framework.md) | Không dùng framework RAG (LangChain/LlamaIndex) cho MVP | Accepted | 2026-11-02 |
| [0004](0004-embedding-bge-m3-via-ollama.md) | Embedding BGE-M3 (dense) qua Ollama | Accepted | 2026-11-02 |
| [0005](0005-vector-store-pgvector-hnsw.md) | PostgreSQL + pgvector với index HNSW cosine | Accepted | 2026-11-02 |
| [0006](0006-chunking-strategy.md) | Chunking: heading-aware + recursive, 512/64 | Accepted | 2026-11-02 |
| [0007](0007-llm-ollama-qwen2-5.md) | LLM generation qua Ollama, mặc định Qwen2.5 7B Instruct | Accepted | 2026-11-02 |
| [0008](0008-local-infra-docker-compose.md) | Hạ tầng local bằng docker compose, app chạy trên host | Accepted | 2026-11-02 |
| [0009](0009-interfaces-cli-and-fastapi.md) | Giao diện CLI (Typer) + REST API (FastAPI) | Accepted | 2026-11-02 |
| [0010](0010-hybrid-search-score-fusion.md) | Hybrid search gộp theo điểm (cosine + α × độ phủ IDF), không dùng RRF | Accepted | 2026-11-03 |
| [0011](0011-reranker-tei-service.md) | Reranker `bge-reranker-v2-m3` trong container TEI riêng (GPU), mặc định tắt | Accepted | 2026-11-03 |
| [0012](0012-mcp-search-docs-for-agents.md) | MCP server `rag mcp` (stdio) cho agent tra cứu tài liệu, chỉ đọc | Accepted (pilot) | 2026-11-09 |
