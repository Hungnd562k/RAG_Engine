# ADR-0012: MCP server `rag mcp` (stdio) cho agent tra cứu tài liệu, chỉ đọc

- **Trạng thái:** Accepted (pilot)
- **Ngày:** 2026-11-09

## Bối cảnh
Mục tiêu của dự án là để agent (Claude Code) tra tài liệu dự án theo ý nghĩa thay vì chỉ grep. Sau M6a, retrieval đã đạt hit@5 = 1.00 trên `samples` và 0.95–1.00 trên E-Wallet. Agent chỉ cần **đoạn tài liệu**, không cần câu trả lời của LLM 3B: Claude tự đọc và tự trả lời. Vì vậy các điểm yếu của `ask` (bịa số, trích dẫn sai) không áp dụng cho đường này.

Agent hỏi khác người: câu ngắn, dạng từ khóa, thường bằng tiếng Anh, kèm định danh. Golden set cũ chỉ có câu hỏi tự nhiên, nên cần đo riêng.

## Quyết định
- **`rag mcp`**: MCP server qua stdio, viết bằng SDK `mcp` 2.x (`MCPServer`). Server chạy trong process của người dùng như REST API, dùng chung `Engine`.
- **Hai tool, chỉ đọc:**
  - `search_docs(query, top_k=5, source_prefix=None)`: trả các chunk kèm **đường dẫn tuyệt đối** (để agent `Read` được), heading, cosine (và điểm rerank nếu bật), nội dung chunk. `top_k` giới hạn ở 1–20.
  - `list_sources(prefix="")`: danh sách file đã ingest.
  - Không có `ask`, `ingest`, `reset`: thay đổi index là việc của người dùng.
- **Độ mới của index:** mỗi file trong kết quả được băm lại (cùng cách băm với loader) và so với hash đã lưu. File đã đổi được đánh dấu `STALE`, file đã xóa được đánh dấu `MISSING`, để agent đọc file thật thay vì tin chunk cũ.
- **Không có kết quả:** trả thông báo có cosine cao nhất và gợi ý dùng grep, không trả danh sách rỗng.
- **Engine khởi tạo lười:** server vẫn khởi động khi Docker chưa chạy. Mỗi lần gọi tool thử kết nối lại; lỗi hạ tầng trả `ToolError` kèm hướng dẫn chuyển sang grep.
- **`instructions` của server** nói rõ khi nào dùng `search_docs` (tìm theo ý nghĩa, không biết file nào, khác ngôn ngữ) và khi nào dùng grep (chuỗi chính xác, cần tìm đủ mọi chỗ).
- **Cấu hình retrieval cho pilot:** giữ mặc định hybrid, không rerank (xem Hệ quả).

## Các phương án đã cân nhắc
| Phương án | Ưu | Nhược |
|-----------|----|-------|
| MCP stdio, chỉ đọc (chọn) | Claude Code hỗ trợ sẵn; không mở port; không cho agent sửa index | Mỗi phiên Claude chạy một process riêng (một pool kết nối Postgres) |
| MCP gọi REST API (`rag serve`) | Một server dùng chung | Phải chạy thêm `rag serve`; thêm một lớp lỗi |
| Agent gọi `rag search --json` qua Bash | Không cần code mới | Phải cấp quyền Bash; khởi động Python mỗi lần (~1 s); không có `instructions` |
| Thêm tool `ingest_docs` | Agent tự làm mới index | Agent ghi vào DB dùng chung; embed trên CPU có thể mất hàng phút |

## Hệ quả
- **Đo trên câu kiểu agent** (`eval/golden-agent.yaml`: 14 + 2 OOS, và bộ E-Wallet riêng tư 20 + 2 OOS), hit@1 / hit@5:

  | Cấu hình | samples (agent) | E-Wallet (agent) | samples | E-Wallet |
  |---|---|---|---|---|
  | hybrid (mặc định) | 0.79 / **1.00** | 0.80 / **1.00** | 1.00 / 1.00 | 0.95 / 0.95 |
  | dense + rerank | 1.00 / 1.00 | 0.75 / 0.95 (mất a14 `recon.daily`) | 1.00 / 1.00 | 0.95 / 1.00 |
  | hybrid + rerank | 1.00 / 1.00 | 0.75 / 1.00 | 1.00 / 1.00 | 0.95 / 0.95 (mất e19) |

  Không cấu hình nào thắng mọi bộ. Agent đọc cả top-5 nên hit@5 quan trọng hơn hit@1; hybrid đạt hit@5 1.00 trên cả hai bộ agent mà không cần thêm service hay VRAM, nên giữ làm mặc định.
- **Tích cực:** search ~10–20 ms, câu ngoài phạm vi kiểu agent (`kubernetes ingress`, `react useEffect`) bị chặn 4/4.
- **Tiêu cực:**
  - Index chỉ mới tới lần `rag ingest` gần nhất; cờ `STALE` chỉ phát hiện file đã đổi, không phát hiện file mới chưa ingest.
  - Source của tài liệu ngoài repo lưu đường dẫn tuyệt đối, nên lọc theo `source_prefix` phải dùng đường dẫn tuyệt đối (agent xem bằng `list_sources`).
  - Phụ thuộc Docker (Postgres, Ollama) đang chạy.
- **Việc cần làm kèm theo:**
  - Đo trong sử dụng thật: agent có gọi `search_docs` đúng lúc không, có đọc file nguồn khi cần không.
  - Collections theo dự án (M6c) thay cho `source_prefix` tuyệt đối.
  - Tự động ingest lại khi tài liệu đổi (hook git hoặc watcher).

## Tham khảo
- [04 §3](../04-interfaces-cli-api.md#3-mcp-server-rag-mcp): tool và cách đăng ký với Claude Code.
- [06 §6.4](../06-evaluation-and-testing.md#64-câu-hỏi-kiểu-agent-mcp-pilot): số đo.
- Model Context Protocol, Python SDK `mcp` 2.x.
