# 05 — Local Setup

Mục tiêu: từ máy trắng (có Docker) → `rag health` OK → `rag search` ra kết quả.

## 1. Yêu cầu
| Thành phần | Phiên bản | Ghi chú |
|------------|-----------|---------|
| Docker + Docker Compose v2 | mới | Chạy Postgres + Ollama |
| uv | ≥ 0.4 | `curl -LsSf https://astral.sh/uv/install.sh \| sh` — cài vào `~/.local/bin` (thêm vào `PATH` nếu chưa có) |
| Python | 3.12 | uv tự tải theo `.python-version`; không dùng 3.14 của hệ thống ([ADR-0002](adr/0002-python-stack-and-tooling.md)) |
| RAM | ≥ 16 GB khuyến nghị | `qwen2.5:3b-instruct` ~1.9 GB, `bge-m3` ~1.2 GB |
| GPU (tùy chọn) | NVIDIA + nvidia-container-toolkit | Không có vẫn chạy CPU; xem [§3](#gpu-nvidia) |
| Ổ đĩa | ~6 GB | Images + models + data |

Máy dev tham chiếu (2026-11-02): RTX 3050 Laptop 4 GB VRAM, 22 GB RAM, 12 luồng CPU, Ubuntu 26.04; đã cài nvidia-container-toolkit, Ollama chạy GPU qua `docker-compose.gpu.yml`.

| Đo trên samples (33 chunks) | CPU | GPU (RTX 3050 4 GB) |
|---|---|---|
| `rag ingest` (model đã nạp) | 6.9 s | 0.6 s |
| Embed 1 query | ~45 ms | ~15 ms |
| Nạp model lần đầu (cold start) | vài giây | ~1.5 s |
| VRAM cho bge-m3 | — | ~0.7 GB (`ollama ps`: 100% GPU) |

## 2. Các bước
```bash
# 1. Infra (docker-compose.yml ở gốc repo)
docker compose up -d
docker compose ps                         # postgres (healthy), ollama (up)

# 2. Models (lần đầu, tải vài GB)
docker compose exec ollama ollama pull bge-m3                # cần từ M1
docker compose exec ollama ollama pull qwen2.5:3b-instruct   # cần từ M3

# 3. Kiểm tra nhanh Ollama embedding
curl -s localhost:11434/api/embed -d '{"model":"bge-m3","input":["xin chào"]}' \
  | python3 -c 'import sys,json; print(len(json.load(sys.stdin)["embeddings"][0]))'   # → 1024

# 4. Python app
cp .env.example .env
uv sync
uv run rag health

# 5. Thử end-to-end (M1–M3)
uv run rag ingest samples/
uv run rag stats
uv run rag search "chính sách nghỉ phép năm"
uv run rag ask "Nhân viên được nghỉ phép bao nhiêu ngày mỗi năm?"

# 6. REST API (M4) — Swagger UI: http://127.0.0.1:8000/docs
uv run rag serve

# 7. Tests
uv run pytest -m "not integration"      # không cần infra
uv run pytest -m integration            # cần bước 1–2; dùng DB riêng rag_test
```

Output `rag health` mong đợi (trước khi pull LLM, dòng LLM là WARN và không làm lệnh fail):
```
 Postgres                 OK    localhost:5432/rag
 pgvector                 OK    v0.8.7
 Schema/meta              OK    bge-m3, dim=1024
 Ollama                   OK    http://localhost:11434
 Embed bge-m3             OK    dim=1024
 LLM qwen2.5:3b-instruct  WARN  chưa pull (cần từ M3): ollama pull qwen2.5:3b-instruct
```

## 3. GPU NVIDIA
Ollama trong container chỉ dùng được GPU khi host có **nvidia-container-toolkit** (cần sudo):
```bash
# Ubuntu — theo hướng dẫn chính thức của NVIDIA Container Toolkit (chạy trong terminal thật, sudo cần nhập mật khẩu)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
docker info | grep -i runtimes            # phải thấy "nvidia"

# Chạy lại Ollama với override GPU
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d
docker compose exec ollama nvidia-smi     # thấy GPU trong container
docker compose exec ollama ollama ps      # cột PROCESSOR = "100% GPU" sau khi gọi embed
```
Với 4 GB VRAM (đã đo ở M3): `bge-m3` và `qwen2.5:3b-instruct` **không cùng vừa** GPU → Ollama nạp/đẩy model liên tục (embed 2–6 s). Cấu hình khuyến nghị: `RAG_EMBED_NUM_GPU=0` (embed trên CPU, ~50 ms/query) để LLM dùng GPU (76% GPU với `RAG_LLM_NUM_CTX=4096`). Kiểm tra bằng `docker compose exec ollama ollama ps`: `bge-m3` = `100% CPU`, `qwen2.5` phần lớn GPU. `qwen2.5:7b-instruct` (~4.7 GB) sẽ bị offload nhiều sang CPU.

**Reranker (M6a, tùy chọn):** service `reranker` (TEI) chỉ đáng bật khi có GPU (bản CPU mất ~12 s cho mỗi câu hỏi).
```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d reranker   # lần đầu tải ~2.2 GB model, warm-up vài chục giây
curl -s localhost:8081/health && echo ok
RAG_RERANK_ENABLED=true uv run rag health                                       # dòng "Reranker OK"
```
**Thứ tự khởi động:** bật reranker **trước** khi LLM được nạp vào GPU. Nếu `qwen2.5` đang chiếm VRAM, TEI không đủ bộ nhớ và thoát với thông báo dễ gây nhầm `cannot find tensor embeddings.word_embeddings.weight`. Cách xử lý: `docker compose exec ollama ollama stop qwen2.5:3b-instruct` rồi `up -d reranker` lại.

Với 4 GB VRAM, reranker chiếm ~2.4 GB nên `qwen2.5:3b` bị đẩy phần lớn sang CPU (generate chậm khoảng 2.5 lần). `search` không bị ảnh hưởng vì không dùng LLM. Xem [ADR-0011](adr/0011-reranker-tei-service.md).

Phương án thay thế: cài Ollama native trên host (dùng GPU trực tiếp), tắt service `ollama` trong compose, giữ `RAG_OLLAMA_URL=http://localhost:11434`.

## 4. `.env.example`
```dotenv
RAG_DATABASE_URL=postgresql://rag:rag@localhost:5432/rag
RAG_OLLAMA_URL=http://localhost:11434
RAG_EMBED_MODEL=bge-m3
RAG_EMBED_DIM=1024
RAG_EMBED_BATCH_SIZE=16
# GPU <= 4 GB: chạy embed trên CPU để nhường VRAM cho LLM (bỏ dòng này nếu GPU đủ lớn)
RAG_EMBED_NUM_GPU=0
RAG_LLM_MODEL=qwen2.5:3b-instruct
RAG_LLM_TEMPERATURE=0.1
RAG_LLM_NUM_CTX=4096
RAG_CHUNK_SIZE=512
RAG_CHUNK_OVERLAP=64
RAG_TOP_K=5
RAG_MIN_SCORE=0.50
# auto = dense khi bật rerank, hybrid khi tắt
RAG_SEARCH_MODE=auto
RAG_HYBRID_ALPHA=0.15
# Bật khi service reranker (GPU) đang chạy
RAG_RERANK_ENABLED=false
RAG_CONTEXT_MAX_CHARS=8000
RAG_LOG_LEVEL=INFO
```
Danh sách đầy đủ biến cấu hình: [01-architecture.md §7](01-architecture.md#7-cấu-hình-biến-môi-trường).

## 5. Troubleshooting
| Triệu chứng | Nguyên nhân | Cách xử lý |
|-------------|-------------|------------|
| `uv: command not found` | `~/.local/bin` chưa có trong PATH | `export PATH="$HOME/.local/bin:$PATH"` (thêm vào `~/.zshrc`) |
| `connection refused :5432` | Postgres chưa healthy / port bị chiếm | `docker compose logs postgres`; đổi port host |
| `Embed bge-m3 FAIL chưa pull` | Chưa pull model | Bước 2 |
| Embed lần đầu chậm vài giây | Ollama load model vào RAM | Bình thường; compose đặt `OLLAMA_KEEP_ALIVE=30m` |
| `Cấu hình lệch với dữ liệu đã lưu` | Đổi `RAG_EMBED_MODEL`/`RAG_EMBED_DIM` sau khi đã ingest | `rag reset --yes` rồi ingest lại |
| `rag search` báo không có chunk đạt ngưỡng | Câu hỏi ngoài tài liệu, hoặc ngưỡng quá cao | Thử `--min-score 0` để xem score thật |
| Lỗi build wheel khi `uv sync` | Đang dùng Python 3.14 | Kiểm tra `.python-version` = 3.12 |
| `rag ask` embed chậm vài giây mỗi câu | Embed và LLM tranh VRAM | `RAG_EMBED_NUM_GPU=0` (xem §3) |
| Reranker thoát với `cannot find tensor embeddings.word_embeddings.weight` | Thiếu VRAM vì LLM đã nạp vào GPU (với image `86-1.8` thì là lỗi thật của image) | `ollama stop qwen2.5:3b-instruct` rồi khởi động lại reranker (xem §3) |
| Tool `search_docs` báo "RAG engine không sẵn sàng" | Docker chưa chạy (vd. sau khi khởi động lại máy) | `docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d` |
| Câu trả lời thiếu `[n]` / lạc đề | Prompt vượt `num_ctx` bị cắt, hoặc model quá nhỏ | Giảm `RAG_CONTEXT_MAX_CHARS`, xem `--show-context`; thử model lớn hơn |
| `rag serve` báo `address already in use` | Port 8000 đang bị chiếm | `uv run rag serve --port 8001` |
| API trả 503 `Engine chưa sẵn sàng` | Server start trước khi compose lên | Bật compose; request tiếp theo engine tự khởi tạo lại |
| Reset sạch hoàn toàn | — | `docker compose down -v` (xóa cả models!) |
