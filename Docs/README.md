# RAG Engine — Bộ tài liệu thiết kế

Tài liệu cho việc xây dựng một RAG engine chạy **local**, tối giản, nhìn thấy kết quả ngay và tự kiểm chứng được.

**Stack chốt:** Python 3.12 · Ollama (`bge-m3` cho embedding, `qwen2.5:7b-instruct` cho generation) · PostgreSQL + pgvector · CLI (Typer) + FastAPI.

## Mục lục

| File | Nội dung | Đọc khi |
|------|----------|---------|
| [00-vision-and-scope.md](00-vision-and-scope.md) | Mục tiêu, phạm vi MVP, tiêu chí thành công | Đầu tiên |
| [01-architecture.md](01-architecture.md) | Components, luồng ingest/query, interface, cấu trúc package | Trước khi code |
| [02-milestones.md](02-milestones.md) | Roadmap M0–M6, Definition of Done + lệnh tự kiểm chứng | Lập kế hoạch / nghiệm thu |
| [03-data-model.md](03-data-model.md) | Schema Postgres/pgvector, index, chiến lược re-ingest | M1 |
| [04-interfaces-cli-api.md](04-interfaces-cli-api.md) | Spec lệnh CLI và REST API | M2–M4 |
| [05-local-setup.md](05-local-setup.md) | Dựng môi trường từ zero, docker compose, `.env` | M0 |
| [06-evaluation-and-testing.md](06-evaluation-and-testing.md) | Chiến lược test, golden set, metrics | M5 |
| [07-risks-and-open-questions.md](07-risks-and-open-questions.md) | Rủi ro và câu hỏi mở | Review định kỳ |
| [08-glossary.md](08-glossary.md) | Thuật ngữ | Khi cần |
| [adr/README.md](adr/README.md) | Architecture Decision Records | Khi cần biết *tại sao* |

## Quy ước
- Viết tiếng Việt, thuật ngữ kỹ thuật giữ tiếng Anh.
- Mọi quyết định kiến trúc có ảnh hưởng lâu dài → ghi ADR mới trong `adr/` (dùng [adr/template.md](adr/template.md)), không sửa ADR đã `Accepted` mà tạo ADR mới `Supersedes`.
- Khi xong một milestone: cập nhật trạng thái trong [02-milestones.md](02-milestones.md).
