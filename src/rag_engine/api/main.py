"""REST API (M4): lớp mỏng bọc engine, schema dùng chung với CLI `--json`.

Chạy: `uv run rag serve` hoặc `uvicorn rag_engine.api.main:app`. Swagger UI: /docs
"""

import logging
import threading
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import psycopg
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from rag_engine import __version__
from rag_engine.config import Settings, get_settings
from rag_engine.embedding.ollama import EmbeddingError
from rag_engine.engine import Engine, build_engine, health_checks
from rag_engine.llm.ollama import LLMError
from rag_engine.schemas import (
    AskRequest,
    AskResponse,
    CheckOut,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    SearchRequest,
    SearchResponse,
    StatsResponse,
)
from rag_engine.store.pgvector import EmbeddingConfigMismatch

log = logging.getLogger(__name__)


def create_app(
    settings: Settings | None = None,
    engine_factory: Callable[[Settings], Engine] = build_engine,
    health_fn: Callable[[Settings], list] = health_checks,
) -> FastAPI:
    settings = settings or get_settings()
    state: dict = {"engine": None, "error": None}
    ingest_lock = threading.Lock()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        # Engine khởi tạo một lần; nếu hạ tầng chưa sẵn sàng, server vẫn chạy để /health báo lỗi.
        try:
            state["engine"] = engine_factory(settings)
        except Exception as exc:  # noqa: BLE001
            state["error"] = exc
            log.error("Không khởi tạo được engine: %s", exc)
        yield
        if state["engine"] is not None:
            state["engine"].close()

    app = FastAPI(
        title="RAG Engine API",
        version=__version__,
        description="Ingest TXT/MD, semantic search và hỏi đáp có trích dẫn. "
        "Xem Docs/04-interfaces-cli-api.md.",
        lifespan=lifespan,
    )

    def get_engine() -> Engine:
        if state["engine"] is None:
            err = state["error"]
            if isinstance(err, EmbeddingConfigMismatch):
                raise HTTPException(409, str(err))
            # Thử lại: hạ tầng có thể đã lên sau khi server khởi động.
            try:
                state["engine"], state["error"] = engine_factory(settings), None
            except EmbeddingConfigMismatch as exc:
                state["error"] = exc
                raise HTTPException(409, str(exc)) from exc
            except Exception as exc:  # noqa: BLE001
                state["error"] = exc
                raise HTTPException(503, f"Engine chưa sẵn sàng: {exc}") from exc
        return state["engine"]

    EngineDep = Annotated[Engine, Depends(get_engine)]

    @app.exception_handler(EmbeddingError)
    @app.exception_handler(LLMError)
    @app.exception_handler(psycopg.OperationalError)
    async def _infra_error(_: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    @app.get("/health", response_model=HealthResponse, tags=["system"])
    def health() -> JSONResponse:
        checks = health_fn(settings)
        required_ok = all(c.ok for c in checks if c.required)
        status = "error" if not required_ok else ("ok" if all(c.ok for c in checks) else "degraded")
        body = HealthResponse(
            status=status,
            embed_model=settings.embed_model,
            llm_model=settings.llm_model,
            checks=[CheckOut(name=c.name, ok=c.ok, required=c.required, detail=c.detail)
                    for c in checks],
        )
        return JSONResponse(status_code=200 if required_ok else 503, content=body.model_dump())

    @app.get("/stats", response_model=StatsResponse, tags=["system"])
    def stats(engine: EngineDep) -> StatsResponse:
        return StatsResponse(**engine.store.stats())

    @app.post("/ingest", response_model=IngestResponse, tags=["ingest"])
    def ingest(req: IngestRequest, engine: EngineDep) -> IngestResponse:
        if not Path(req.path).exists():
            raise HTTPException(400, f"Không tìm thấy đường dẫn: {req.path}")
        if not ingest_lock.acquire(blocking=False):
            raise HTTPException(409, "Đang có một ingest khác chạy, thử lại sau")
        try:
            report = engine.pipeline.ingest(req.path, prune=req.prune, dry_run=req.dry_run)
        finally:
            ingest_lock.release()
        return IngestResponse.of(report, dry_run=req.dry_run)

    @app.post("/search", response_model=SearchResponse, tags=["query"])
    def search(req: SearchRequest, engine: EngineDep) -> SearchResponse:
        try:
            res = engine.retriever.retrieve(
                req.query, req.top_k or settings.top_k,
                source_prefix=req.source_prefix, min_score=req.min_score,
            )
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return SearchResponse.of(res)

    @app.post("/ask", response_model=AskResponse, tags=["query"])
    def ask(req: AskRequest, engine: EngineDep) -> AskResponse:
        try:
            ans = engine.orchestrator.ask(
                req.question, req.top_k or settings.top_k,
                source_prefix=req.source_prefix, min_score=req.min_score,
            )
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return AskResponse.of(ans)

    return app


def __getattr__(name: str):
    # `uvicorn rag_engine.api.main:app` — tạo app khi được truy cập, không phải lúc import module.
    if name == "app":
        return create_app()
    raise AttributeError(name)
