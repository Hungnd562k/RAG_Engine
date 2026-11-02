# ADR-0002: Python 3.12 + uv và bộ thư viện nền

- **Trạng thái:** Accepted
- **Ngày:** 2026-11-02

## Bối cảnh
Source code viết bằng Python. Máy dev có Python 3.14 hệ thống — quá mới, một số thư viện (đặc biệt có phần native) có thể chưa có wheel. Cần môi trường tái lập được, cài nhanh, ít công cụ.

## Quyết định
- **Python 3.12**, pin bằng `.python-version`; quản lý bằng **uv** (venv, lockfile `uv.lock`, chạy lệnh `uv run`).
- Layout `src/rag_engine/`, entrypoint script `rag`.
- Thư viện nền:

| Mục đích | Thư viện |
|----------|----------|
| Config | `pydantic-settings` |
| Data models / API schema | `pydantic` v2, `dataclasses` |
| Postgres driver | `psycopg[binary,pool]` 3 + `pgvector` (adapter Python) |
| HTTP client tới Ollama | `httpx` (gọi REST trực tiếp, không cần SDK) |
| CLI | `typer` + `rich` |
| API | `fastapi` + `uvicorn` |
| Eval data | `pyyaml` |
| Test | `pytest` (+ marker `integration`) |
| Lint/format | `ruff` |

## Các phương án đã cân nhắc
| Phương án | Ưu | Nhược |
|-----------|----|-------|
| uv + 3.12 (chọn) | Nhanh, lockfile, tự tải Python | Công cụ tương đối mới |
| Poetry | Phổ biến | Chậm hơn, không quản lý phiên bản Python |
| pip + venv | Không cần cài thêm | Không lockfile chuẩn |
| Python 3.14 | Mới nhất | Rủi ro thiếu wheel |
| SQLAlchemy ORM | Trừu tượng hóa DB | Thừa cho 3 bảng, khó dùng toán tử vector trực tiếp |

## Hệ quả
- SQL viết tay trong `store/pgvector.py`, rõ ràng, dễ kiểm chứng.
- Gọi Ollama qua `httpx` giúp dễ mock trong unit test.
- Người mới chỉ cần `uv sync` để có môi trường.
