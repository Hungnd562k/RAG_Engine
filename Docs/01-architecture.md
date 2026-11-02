# 01 — Kiến trúc

## 1. Tổng quan

```mermaid
flowchart LR
    subgraph Clients
        CLI[CLI - Typer]
        API[REST API - FastAPI]
    end

    subgraph Engine["rag_engine (Python)"]
        CFG[Config]
        ORCH[RAG Orchestrator]
        subgraph Ingestion
            LOAD[Document Loader]
            CHUNK[Chunker]
            IPIPE[Ingestion Pipeline]
        end
        subgraph Query
            RET[Retriever]
            PROMPT[Prompt Builder]
            GEN[Generator / LLMProvider]
        end
        EMB[Embedder]
        VS[Vector Store]
        EVAL[Evaluator]
    end

    subgraph Infra["Infra (docker compose)"]
        PG[(PostgreSQL + pgvector)]
        OLL[Ollama: bge-m3, qwen2.5:3b-instruct]
    end

    CLI --> ORCH
    CLI --> IPIPE
    CLI --> EVAL
    API --> ORCH
    API --> IPIPE
    IPIPE --> LOAD --> CHUNK --> EMB
    IPIPE --> VS
    ORCH --> RET --> EMB
    RET --> VS
    ORCH --> PROMPT
    ORCH --> GEN
    EVAL --> RET
    EMB --> OLL
    GEN --> OLL
    VS --> PG
    CFG -.-> Engine
```

## 2. Danh sách components

| # | Component | Trách nhiệm | Input → Output | Milestone |
|---|-----------|-------------|----------------|-----------|
| 1 | **Config** | Đọc `.env`/biến môi trường bằng `pydantic-settings`, validate | env → `Settings` | M0 |
| 2 | **Document Loader** | Duyệt file/thư mục, đọc TXT/MD (UTF-8), tính `content_hash` | path → `list[Document]` | M1 |
| 3 | **Chunker** | Tách theo heading Markdown, rồi recursive split theo đoạn/câu, overlap; giữ `heading_path` | `Document` → `list[Chunk]` | M1 |
| 4 | **Embedder** | Gọi Ollama `/api/embed` theo batch, kiểm tra dim = 1024 | `list[str]` → `list[list[float]]` | M1 |
| 5 | **Vector Store** | Schema, upsert document/chunks trong transaction, similarity search, stats, reset | chunks + vectors ↔ Postgres | M1–M2 |
| 6 | **Ingestion Pipeline** | load → so hash → chunk → embed → upsert; báo cáo added/updated/skipped | path → `IngestReport` | M1 |
| 7 | **Retriever** | Embed query → top-k cosine, ngưỡng `min_score`, filter theo `source` | query → `list[RetrievedChunk]` | M2 |
| 8 | **Prompt Builder** | Ghép system prompt + context đánh số `[1]..[n]` + câu hỏi; cắt theo budget ký tự | chunks + query → messages | M3 |
| 9 | **Generator (LLMProvider)** | Interface chat; adapter `OllamaLLM` gọi `/api/chat` | messages → text | M3 |
| 10 | **RAG Orchestrator** | `search()` và `ask()`; đo timing từng bước; trả sources | query → `Answer` | M3 |
| 11 | **CLI** | Lệnh người dùng; hiển thị bảng bằng `rich` | — | M0–M5 |
| 12 | **REST API** | FastAPI bọc Orchestrator/Pipeline; Swagger tại `/docs` | HTTP JSON | M4 |
| 13 | **Evaluator** | Chạy golden set, tính hit@k, MRR | YAML → report | M5 |
| 14 | **Observability** | Logging chuẩn (`logging` + format key=value), timing | — | M0+ |

## 3. Luồng Ingest

```mermaid
sequenceDiagram
    participant U as CLI/API
    participant P as IngestionPipeline
    participant L as Loader
    participant C as Chunker
    participant E as Embedder
    participant V as VectorStore
    U->>P: ingest(path)
    P->>L: load(path)
    L-->>P: documents (text, source, hash)
    loop mỗi document
        P->>V: get_document_hash(source)
        alt hash không đổi
            P-->>P: skip
        else mới hoặc đã đổi
            P->>C: split(document)
            C-->>P: chunks
            P->>E: embed([chunk.text]) theo batch
            E-->>P: vectors
            P->>V: replace_document(doc, chunks, vectors) (1 transaction)
        end
    end
    P-->>U: IngestReport(added, updated, skipped, chunks, elapsed)
```

## 4. Luồng Query (`ask`)

```mermaid
sequenceDiagram
    participant U as CLI/API
    participant O as Orchestrator
    participant R as Retriever
    participant E as Embedder
    participant V as VectorStore
    participant B as PromptBuilder
    participant G as LLMProvider
    U->>O: ask(question, top_k)
    O->>R: retrieve(question, top_k)
    R->>E: embed([question])
    R->>V: search(vector, top_k, min_score)
    V-->>R: chunks + score
    R-->>O: retrieved
    alt không có chunk đạt ngưỡng
        O-->>U: "Không tìm thấy thông tin trong tài liệu"
    else
        O->>B: build(question, retrieved)
        O->>G: chat(messages)
        G-->>O: answer text có [n]
        O-->>U: Answer(text, sources, timings)
    end
```

## 5. Interface chính (Python, phác thảo)

```python
@dataclass
class Document:
    source: str            # đường dẫn tương đối, duy nhất
    text: str
    content_hash: str      # sha256
    metadata: dict

@dataclass
class Chunk:
    index: int
    text: str
    heading_path: list[str]
    char_start: int
    char_end: int

@dataclass
class RetrievedChunk:
    chunk_id: int
    source: str
    text: str
    heading_path: list[str]
    score: float           # cosine similarity, 1 = giống nhất

class Embedder(Protocol):
    dim: int
    def embed(self, texts: list[str]) -> list[list[float]]: ...

class VectorStore(Protocol):
    def get_document_hash(self, source: str) -> str | None: ...
    def replace_document(self, doc: Document, chunks: list[Chunk], vectors: list[list[float]]) -> None: ...
    def search(self, vector: list[float], top_k: int, min_score: float = 0.0,
               source_prefix: str | None = None) -> list[RetrievedChunk]: ...
    def stats(self) -> dict: ...
    def reset(self) -> None: ...

class LLMProvider(Protocol):
    def chat(self, messages: list[dict], temperature: float = 0.1) -> str: ...

@dataclass
class Answer:
    text: str
    sources: list[RetrievedChunk]
    timings_ms: dict[str, float]   # embed, search, generate, total
```

## 6. Cấu trúc package đề xuất

```
RAG_Engine/
├── docker-compose.yml
├── pyproject.toml                # uv, Python 3.12
├── .env.example
├── src/rag_engine/
│   ├── config.py
│   ├── models.py                 # Document, Chunk, RetrievedChunk, Answer
│   ├── ingestion/
│   │   ├── loader.py
│   │   ├── chunker.py
│   │   └── pipeline.py
│   ├── embedding/
│   │   ├── base.py               # Embedder Protocol
│   │   └── ollama.py
│   ├── store/
│   │   ├── base.py               # VectorStore Protocol
│   │   ├── pgvector.py
│   │   └── schema.sql
│   ├── llm/
│   │   ├── base.py               # LLMProvider Protocol
│   │   └── ollama.py
│   ├── retrieval/retriever.py
│   ├── generation/prompt.py
│   ├── orchestrator.py
│   ├── evaluation/evaluator.py
│   ├── cli.py                    # entrypoint `rag`
│   └── api/main.py               # FastAPI app
├── samples/                      # tài liệu mẫu để tự kiểm chứng
├── eval/golden.yaml
└── tests/
    ├── unit/
    └── integration/
```

Phụ thuộc chỉ đi một chiều: `cli`/`api` → `orchestrator`/`pipeline` → các module lá (`embedding`, `store`, `llm`, `ingestion`). Module lá không import lẫn nhau, chỉ phụ thuộc `models` và `config`. Việc lắp ráp (wiring) nằm ở một hàm factory `build_engine(settings)`.

## 7. Cấu hình (biến môi trường)

| Biến | Mặc định | Ghi chú |
|------|----------|---------|
| `RAG_DATABASE_URL` | `postgresql://rag:rag@localhost:5432/rag` | |
| `RAG_OLLAMA_URL` | `http://localhost:11434` | |
| `RAG_EMBED_MODEL` | `bge-m3` | |
| `RAG_EMBED_DIM` | `1024` | Phải khớp schema |
| `RAG_EMBED_BATCH_SIZE` | `16` | |
| `RAG_OLLAMA_TIMEOUT_S` | `300` | Timeout HTTP tới Ollama |
| `RAG_LLM_MODEL` | `qwen2.5:3b-instruct` | Vừa GPU 4 GB; 7B dùng được nếu đủ VRAM (xem ADR-0007 addendum) |
| `RAG_LLM_TEMPERATURE` | `0.1` | |
| `RAG_CHUNK_SIZE` | `512` | token ước lượng = ký tự / `RAG_CHARS_PER_TOKEN` |
| `RAG_CHUNK_OVERLAP` | `64` | |
| `RAG_CHARS_PER_TOKEN` | `4.0` | Hệ số ước lượng token |
| `RAG_MIN_SECTION_TOKENS` | `50` | Section nhỏ hơn được gộp vào section sau |
| `RAG_TOP_K` | `5` | |
| `RAG_MIN_SCORE` | `0.50` | Ngưỡng cosine, đã đo trên samples ([Q6](07-risks-and-open-questions.md#q6--ngưỡng-min_score)) |
| `RAG_HNSW_EF_SEARCH` | `40` | Tăng để recall cao hơn |
| `RAG_CONTEXT_MAX_CHARS` | `12000` | Budget context đưa vào prompt |
| `RAG_LOG_LEVEL` | `INFO` | |

## 8. Prompt mặc định (M3)

```
SYSTEM:
Bạn là trợ lý trả lời dựa trên tài liệu được cung cấp.
- Chỉ sử dụng thông tin trong phần CONTEXT.
- Mỗi ý phải kèm trích dẫn dạng [n] tương ứng với đoạn context.
- Nếu CONTEXT không đủ để trả lời, hãy nói: "Không tìm thấy thông tin trong tài liệu."
- Trả lời bằng ngôn ngữ của câu hỏi.

USER:
CONTEXT:
[1] (source: docs/a.md › Heading A)
...
[2] (source: docs/b.md › Heading B)
...

QUESTION: <câu hỏi>
```

## 9. Điểm mở rộng (M6+)
- **Hybrid search:** thêm cột `tsvector` + GIN index, kết hợp với vector bằng Reciprocal Rank Fusion trong `Retriever`.
- **Reranker:** bước `Reranker` (vd. `bge-reranker-v2-m3`) giữa Retriever và PromptBuilder.
- **Loaders mới:** đăng ký theo đuôi file (`.pdf`, `.docx`, `.html`).
- **Collections:** thêm cột `collection` để tách namespace.
- **LLM khác:** adapter mới implement `LLMProvider` (Claude, OpenAI…), chọn qua config.
