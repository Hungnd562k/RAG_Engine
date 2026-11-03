"""Wiring: lắp ráp các component từ Settings. CLI/API chỉ gọi qua đây."""

from dataclasses import dataclass

import httpx

from rag_engine.config import Settings
from rag_engine.embedding.ollama import OllamaEmbedder
from rag_engine.generation.prompt import PromptBuilder
from rag_engine.ingestion.chunker import Chunker
from rag_engine.ingestion.pipeline import IngestionPipeline
from rag_engine.llm.ollama import OllamaLLM
from rag_engine.orchestrator import RagOrchestrator
from rag_engine.rerank.tei import TEIReranker
from rag_engine.retrieval.retriever import Retriever
from rag_engine.store.pgvector import PgVectorStore


@dataclass
class Engine:
    settings: Settings
    embedder: OllamaEmbedder
    store: PgVectorStore
    chunker: Chunker
    pipeline: IngestionPipeline
    retriever: Retriever
    llm: OllamaLLM
    orchestrator: RagOrchestrator

    def close(self) -> None:
        self.store.close()


def build_embedder(s: Settings) -> OllamaEmbedder:
    return OllamaEmbedder(
        base_url=s.ollama_url,
        model=s.embed_model,
        dim=s.embed_dim,
        batch_size=s.embed_batch_size,
        timeout_s=s.ollama_timeout_s,
        num_gpu=s.embed_num_gpu,
    )


def build_store(s: Settings, verify_meta: bool = True) -> PgVectorStore:
    return PgVectorStore(
        s.database_url, s.embed_model, s.embed_dim, s.hnsw_ef_search, verify_meta=verify_meta
    )


def build_reranker(s: Settings) -> TEIReranker | None:
    return TEIReranker(s.rerank_url, s.rerank_timeout_s) if s.rerank_enabled else None


def build_engine(s: Settings) -> Engine:
    embedder = build_embedder(s)
    store = build_store(s)
    chunker = Chunker(s.chunk_size, s.chunk_overlap, s.chars_per_token, s.min_section_tokens)
    retriever = Retriever(
        embedder, store, s.min_score,
        mode=s.effective_search_mode, alpha=s.hybrid_alpha, candidates=s.hybrid_candidates,
        reranker=build_reranker(s), rerank_candidates=s.rerank_candidates,
    )
    llm = OllamaLLM(
        base_url=s.ollama_url,
        model=s.llm_model,
        temperature=s.llm_temperature,
        num_ctx=s.llm_num_ctx,
        timeout_s=s.ollama_timeout_s,
    )
    return Engine(
        settings=s,
        embedder=embedder,
        store=store,
        chunker=chunker,
        pipeline=IngestionPipeline(chunker, embedder, store),
        retriever=retriever,
        llm=llm,
        orchestrator=RagOrchestrator(retriever, PromptBuilder(s.context_max_chars), llm),
    )


@dataclass
class Check:
    name: str
    ok: bool
    detail: str
    required: bool = True


def _model_present(name: str, available: list[str]) -> bool:
    return name in available or (":" not in name and f"{name}:latest" in available)


def health_checks(s: Settings) -> list[Check]:
    """Kiểm tra DB, pgvector, Ollama, models. Không raise; trả về danh sách kết quả."""
    checks: list[Check] = []

    try:
        store = build_store(s, verify_meta=False)
        checks.append(Check("Postgres", True, s.database_url.rsplit("@", 1)[-1]))
        checks.append(Check("pgvector", True, f"v{store.extension_version()}"))
        try:
            store.check_meta()
            checks.append(Check("Schema/meta", True, f"{s.embed_model}, dim={s.embed_dim}"))
        except Exception as exc:  # noqa: BLE001
            checks.append(Check("Schema/meta", False, str(exc)))
        store.close()
    except Exception as exc:  # noqa: BLE001
        checks.append(Check("Postgres", False, f"{type(exc).__name__}: {exc}".strip()[:200]))

    if s.rerank_enabled:
        try:
            info = TEIReranker(s.rerank_url, timeout_s=5).info()
            checks.append(Check("Reranker", True, f"{info.get('model_id')} @ {s.rerank_url}"))
        except httpx.HTTPError as exc:
            checks.append(Check("Reranker", False, f"{s.rerank_url}: {exc} (docker compose up -d reranker)"))
    else:
        checks.append(Check("Reranker", True, "tắt (RAG_RERANK_ENABLED=false)", required=False))

    embedder = build_embedder(s)
    try:
        available = embedder.list_models()
        checks.append(Check("Ollama", True, s.ollama_url))
    except httpx.HTTPError as exc:
        checks.append(Check("Ollama", False, f"{s.ollama_url}: {exc}"))
        return checks

    if _model_present(s.embed_model, available):
        try:
            dim = len(embedder.embed(["ping"])[0])
            checks.append(Check(f"Embed {s.embed_model}", True, f"dim={dim}"))
        except Exception as exc:  # noqa: BLE001
            checks.append(Check(f"Embed {s.embed_model}", False, str(exc)[:200]))
    else:
        checks.append(
            Check(f"Embed {s.embed_model}", False, f"chưa pull: ollama pull {s.embed_model}")
        )

    llm_ok = _model_present(s.llm_model, available)
    checks.append(
        Check(
            f"LLM {s.llm_model}",
            llm_ok,
            "có sẵn" if llm_ok else f"chưa pull (cần từ M3): ollama pull {s.llm_model}",
            required=False,
        )
    )
    return checks
