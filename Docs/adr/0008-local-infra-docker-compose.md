# ADR-0008: Hạ tầng local bằng docker compose, app chạy trên host

- **Trạng thái:** Accepted
- **Ngày:** 2026-11-02

## Bối cảnh
Máy dev đã có Docker nhưng chưa cài Ollama và Postgres. Cần dựng môi trường nhanh, tái lập được, dễ dọn dẹp. Trong lúc phát triển, cần vòng lặp sửa code – chạy lại nhanh.

## Quyết định
- `docker-compose.yml` gồm 2 service: `postgres` (`pgvector/pgvector:pg16`) và `ollama` (`ollama/ollama`), volume riêng cho data và model; GPU passthrough là tùy chọn.
- **App Python chạy trên host** bằng `uv run` (không container hóa ở MVP) để debug/hot-reload dễ.
- Đóng gói app thành image + thêm service `app` vào compose để ở M6+.

## Các phương án đã cân nhắc
| Phương án | Ưu | Nhược |
|-----------|----|-------|
| Compose cho infra, app trên host (chọn) | Tái lập infra, dev loop nhanh | Cần uv trên host |
| Cài Ollama/Postgres native | Hiệu năng GPU trực tiếp | Khó tái lập, khó dọn |
| Mọi thứ trong container | Giống production | Dev loop chậm hơn, debug khó hơn |

## Hệ quả
- Một lệnh `docker compose up -d` dựng toàn bộ phụ thuộc.
- Trên máy có GPU NVIDIA cần `nvidia-container-toolkit` để Ollama trong container dùng GPU; nếu không, chấp nhận chạy CPU hoặc cài Ollama native và trỏ `RAG_OLLAMA_URL` tới đó.
