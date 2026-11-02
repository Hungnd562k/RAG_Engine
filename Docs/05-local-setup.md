# 05 — Local Setup

Mục tiêu: từ máy trắng (có Docker) → `rag health` OK.

## 1. Yêu cầu
| Thành phần | Phiên bản | Ghi chú |
|------------|-----------|---------|
| Docker + Docker Compose v2 | mới | Chạy Postgres + Ollama |
| uv | ≥ 0.4 | Quản lý Python & dependencies (`curl -LsSf https://astral.sh/uv/install.sh \| sh`) |
| Python | 3.12 | uv tự tải; không dùng 3.14 của hệ thống ([ADR-0002](adr/0002-python-stack-and-tooling.md)) |
| RAM | ≥ 16 GB khuyến nghị | `qwen2.5:7b` (Q4) ~5 GB, `bge-m3` ~1.2 GB |
| GPU (tùy chọn) | NVIDIA + nvidia-container-toolkit | Không có GPU vẫn chạy, generation chậm hơn (vài chục giây/câu) |
| Ổ đĩa | ~10 GB | Models + data |

## 2. docker-compose.yml (sẽ tạo ở M0)
```yaml
services:
  postgres:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_USER: rag
      POSTGRES_PASSWORD: rag
      POSTGRES_DB: rag
    ports: ["5432:5432"]
    volumes: [pgdata:/var/lib/postgresql/data]
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U rag"]
      interval: 5s
      retries: 10

  ollama:
    image: ollama/ollama:latest
    ports: ["11434:11434"]
    volumes: [ollama:/root/.ollama]
    # Bỏ comment nếu có GPU NVIDIA:
    # deploy:
    #   resources:
    #     reservations:
    #       devices: [{ driver: nvidia, count: all, capabilities: [gpu] }]

volumes:
  pgdata:
  ollama:
```

## 3. Các bước
```bash
# 1. Infra
docker compose up -d
docker compose ps                         # postgres healthy, ollama up

# 2. Models (lần đầu, tải vài GB)
docker compose exec ollama ollama pull bge-m3
docker compose exec ollama ollama pull qwen2.5:7b-instruct

# 3. Kiểm tra nhanh Ollama embedding
curl -s localhost:11434/api/embed -d '{"model":"bge-m3","input":["xin chào"]}' \
  | python3 -c 'import sys,json; print(len(json.load(sys.stdin)["embeddings"][0]))'   # → 1024

# 4. Python app
cp .env.example .env
uv sync
uv run rag health

# 5. Thử end-to-end
uv run rag ingest samples/
uv run rag search "nghỉ phép"
uv run rag ask "Nhân viên được nghỉ phép bao nhiêu ngày?"
```

## 4. `.env.example`
```dotenv
RAG_DATABASE_URL=postgresql://rag:rag@localhost:5432/rag
RAG_OLLAMA_URL=http://localhost:11434
RAG_EMBED_MODEL=bge-m3
RAG_EMBED_DIM=1024
RAG_EMBED_BATCH_SIZE=16
RAG_LLM_MODEL=qwen2.5:7b-instruct
RAG_LLM_TEMPERATURE=0.1
RAG_CHUNK_SIZE=512
RAG_CHUNK_OVERLAP=64
RAG_TOP_K=5
RAG_MIN_SCORE=0.35
RAG_CONTEXT_MAX_CHARS=12000
RAG_LOG_LEVEL=INFO
```

## 5. Troubleshooting
| Triệu chứng | Nguyên nhân | Cách xử lý |
|-------------|-------------|------------|
| `connection refused :5432` | Postgres chưa healthy / port bị chiếm | `docker compose logs postgres`; đổi port host |
| `model "bge-m3" not found` | Chưa pull model | Bước 2 |
| Embed lần đầu rất chậm | Ollama load model vào RAM | Bình thường; các lần sau nhanh. Có thể đặt `OLLAMA_KEEP_ALIVE=30m` |
| `rag ask` > 60s | Chạy CPU, model 7B | Dùng GPU hoặc model nhỏ hơn: `qwen2.5:3b-instruct` |
| `expected 1024 dimensions` | Đổi embed model mà không reset | `rag reset --yes` rồi ingest lại |
| Lỗi build wheel khi `uv sync` | Đang dùng Python 3.14 | Kiểm tra `.python-version` = 3.12 |
| Reset sạch hoàn toàn | — | `docker compose down -v` (xóa cả models!) |
