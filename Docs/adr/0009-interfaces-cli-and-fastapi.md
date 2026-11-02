# ADR-0009: Giao diện CLI (Typer) + REST API (FastAPI)

- **Trạng thái:** Accepted
- **Ngày:** 2026-11-02

## Bối cảnh
Yêu cầu "nhìn thấy ngay kết quả và tự kiểm chứng" → cần cách tương tác nhanh nhất cho developer. Mục tiêu dài hạn là engine dùng lại được bởi nhiều ứng dụng → cần API.

## Quyết định
- **CLI `rag`** (Typer + rich) là giao diện chính cho M0–M3: hiển thị bảng kết quả, score, nguồn, timing; có `--json`.
- **FastAPI** ở M4, bọc cùng engine; Swagger UI `/docs` thay cho UI riêng.
- Cả hai chỉ là lớp mỏng gọi `build_engine(settings)`; không chứa logic nghiệp vụ.
- Chưa làm web UI chat ở MVP.

## Các phương án đã cân nhắc
| Phương án | Ưu | Nhược |
|-----------|----|-------|
| CLI + FastAPI (chọn) | Nhanh để thử, scriptable; API sẵn cho tích hợp | Không có giao diện chat đẹp |
| Chỉ CLI | Tối giản nhất | Chưa tích hợp được với app khác |
| Streamlit/Gradio UI | Trực quan | Thêm dependency, lệch mục tiêu engine |

## Hệ quả
- Output `--json` của CLI và response API dùng chung Pydantic schema → so sánh trực tiếp được (DoD của M4).
- Web UI có thể thêm sau như một client của API.
