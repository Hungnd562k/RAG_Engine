"""Schema JSON dùng chung cho REST API và `rag ... --json` (Docs/04-interfaces-cli-api.md)."""

from pydantic import BaseModel, Field

from rag_engine.models import Answer, IngestReport, RetrievedChunk, SearchResult

# ---------- requests ----------


class IngestRequest(BaseModel):
    path: str = Field(description="Đường dẫn file/thư mục TXT/MD trên máy chạy server")
    prune: bool = False
    dry_run: bool = False


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int | None = Field(None, gt=0, le=50)
    source_prefix: str | None = None
    min_score: float | None = Field(None, ge=-1.0, le=1.0)


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int | None = Field(None, gt=0, le=50)
    source_prefix: str | None = None
    min_score: float | None = Field(None, ge=-1.0, le=1.0)


# ---------- responses ----------


class ChunkOut(BaseModel):
    chunk_id: int
    source: str
    heading_path: list[str]
    score: float
    text: str

    @classmethod
    def of(cls, c: RetrievedChunk) -> "ChunkOut":
        return cls(chunk_id=c.chunk_id, source=c.source, heading_path=list(c.heading_path),
                   score=c.score, text=c.text)


class SourceOut(ChunkOut):
    ref: int


class SearchResponse(BaseModel):
    results: list[ChunkOut]
    timings_ms: dict[str, float]

    @classmethod
    def of(cls, r: SearchResult) -> "SearchResponse":
        return cls(results=[ChunkOut.of(c) for c in r.results], timings_ms=r.timings_ms)


class AskResponse(BaseModel):
    answer: str
    answered: bool = Field(description="False khi bị chặn bởi min_score (không gọi LLM)")
    sources: list[SourceOut]
    cited: list[int] = Field(description="Các [n] hợp lệ xuất hiện trong answer")
    invalid_citations: list[int] = Field(description="Các [n] không trỏ tới nguồn nào")
    timings_ms: dict[str, float]

    @classmethod
    def of(cls, a: Answer) -> "AskResponse":
        return cls(
            answer=a.text,
            answered=a.answered,
            sources=[SourceOut(ref=i, **ChunkOut.of(s).model_dump()) for i, s in enumerate(a.sources, 1)],
            cited=a.cited,
            invalid_citations=a.invalid_citations,
            timings_ms=a.timings_ms,
        )


class IngestResponse(BaseModel):
    added: int
    updated: int
    skipped: int
    deleted: int
    chunks: int
    elapsed_ms: float
    dry_run: bool
    details: list[dict]

    @classmethod
    def of(cls, r: IngestReport, dry_run: bool = False) -> "IngestResponse":
        return cls(
            added=r.added, updated=r.updated, skipped=r.skipped, deleted=r.deleted,
            chunks=r.chunks, elapsed_ms=r.elapsed_ms, dry_run=dry_run,
            details=[{"source": s, "action": a, "chunks": n} for s, a, n in r.details],
        )


class StatsResponse(BaseModel):
    documents: int
    chunks: int
    embed_model: str | None = None
    embed_dim: int | None = None
    pgvector: str | None = None


class CheckOut(BaseModel):
    name: str
    ok: bool
    required: bool
    detail: str


class HealthResponse(BaseModel):
    status: str = Field(description="ok | degraded (thiếu thành phần không bắt buộc) | error")
    embed_model: str
    llm_model: str
    checks: list[CheckOut]
