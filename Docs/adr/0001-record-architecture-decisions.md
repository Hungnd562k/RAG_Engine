# ADR-0001: Ghi lại quyết định kiến trúc bằng ADR

- **Trạng thái:** Accepted
- **Ngày:** 2026-11-02

## Bối cảnh
RAG engine có nhiều lựa chọn kỹ thuật dễ thay đổi theo thời gian (embedding model, vector DB, LLM, chunking). Khi quay lại sau vài tháng, cần biết *vì sao* đã chọn như vậy để quyết định có nên thay hay không.

## Quyết định
Dùng Architecture Decision Records dạng MADR rút gọn, lưu trong `Docs/adr/`, đánh số tăng dần, theo [template.md](template.md). Mỗi ADR một quyết định. ADR đã `Accepted` là bất biến; thay đổi bằng ADR mới đánh dấu `Supersedes`.

## Các phương án đã cân nhắc
| Phương án | Ưu | Nhược |
|-----------|----|-------|
| ADR markdown trong repo (chọn) | Cùng version với code, review qua PR | Cần kỷ luật cập nhật |
| Wiki/Confluence | Dễ đọc cho người ngoài | Tách rời code, dễ lỗi thời |
| Không ghi | Nhanh | Mất ngữ cảnh quyết định |

## Hệ quả
- Mỗi thay đổi kiến trúc đáng kể (đổi model, store, schema lớn) kèm một ADR.
- Index ADR duy trì trong [README.md](README.md).
