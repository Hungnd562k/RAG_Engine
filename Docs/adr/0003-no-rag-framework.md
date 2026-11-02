# ADR-0003: Không dùng framework RAG (LangChain/LlamaIndex) cho MVP

- **Trạng thái:** Accepted
- **Ngày:** 2026-11-02

## Bối cảnh
Mục tiêu MVP là *hiểu rõ và tự kiểm chứng* từng bước của RAG. Các framework như LangChain/LlamaIndex rút ngắn thời gian dựng nhưng che đi chi tiết (chunking, prompt, truy vấn vector), API thay đổi nhanh, kéo theo nhiều dependency.

## Quyết định
Tự viết các module mỏng (loader, chunker, embedder, store, retriever, prompt, llm, orchestrator) với interface `Protocol` rõ ràng. Tổng code lõi dự kiến < 1.500 dòng. Có thể tham khảo/vay mượn thuật toán (vd. recursive splitter) nhưng không phụ thuộc framework.

## Các phương án đã cân nhắc
| Phương án | Ưu | Nhược |
|-----------|----|-------|
| Tự viết lớp mỏng (chọn) | Minh bạch, ít dependency, dễ test, kiểm soát SQL/prompt | Tự viết phần tiện ích, chậm hơn lúc đầu |
| LangChain | Hệ sinh thái lớn, nhiều integration | Trừu tượng nặng, breaking changes thường xuyên, khó debug |
| LlamaIndex | Mạnh về indexing/retrieval | Tương tự, nhiều khái niệm phải học |
| Haystack | Pipeline rõ ràng | Vẫn là framework lớn so với nhu cầu |

## Hệ quả
- Engine trở thành thư viện tái sử dụng có API riêng ổn định.
- Khi cần tính năng nâng cao (agent, nhiều loader) có thể viết adapter để dùng thành phần của framework sau, mà không đổi lõi.
