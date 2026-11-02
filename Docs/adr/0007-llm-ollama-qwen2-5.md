# ADR-0007: LLM generation qua Ollama, mặc định Qwen2.5 7B Instruct

- **Trạng thái:** Accepted
- **Ngày:** 2026-11-02

## Bối cảnh
Bước generation cần LLM chạy local (offline, không phí), trả lời tốt tiếng Việt, tuân thủ chỉ dẫn trích dẫn. Ollama đã có sẵn trong stack cho embedding.

## Quyết định
- Định nghĩa `LLMProvider` Protocol (`chat(messages, temperature) -> str`); adapter đầu tiên `OllamaLLM` gọi `POST /api/chat` với `stream: false`.
- Model mặc định: **`qwen2.5:7b-instruct`** (đa ngôn ngữ tốt, gồm tiếng Việt; context 32k). Phương án nhẹ: `qwen2.5:3b-instruct` cho máy không GPU.
- `temperature = 0.1`, `num_ctx` đủ chứa context (vd. 8192).
- Model chọn bằng `RAG_LLM_MODEL`, không hardcode.

## Các phương án đã cân nhắc
| Phương án | Ưu | Nhược |
|-----------|----|-------|
| Qwen2.5 7B Instruct (chọn) | Tiếng Việt tốt, tuân thủ instruction tốt, phổ biến trên Ollama | ~5 GB RAM, chậm trên CPU |
| Llama 3.1 8B Instruct | Phổ biến, mạnh tiếng Anh | Tiếng Việt kém hơn Qwen |
| Gemma 2 9B | Chất lượng tốt | Nặng hơn, context 8k |
| Model 1–3B | Nhanh | Hay bỏ qua chỉ dẫn trích dẫn |
| Claude / OpenAI API | Chất lượng cao nhất | Cần internet, phí; để sau qua adapter mới |

## Hệ quả
- Engine hoàn toàn offline cho MVP.
- Model mặc định có thể thay đổi khi có model mới tốt hơn trên Ollama — chỉ cần đổi config, không cần ADR mới trừ khi đổi provider.
- Thêm provider cloud về sau = thêm một adapter, không đụng orchestrator.
