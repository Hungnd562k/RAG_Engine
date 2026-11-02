"""API tests với engine giả (không cần Postgres/Ollama)."""

import pytest
from fastapi.testclient import TestClient

from rag_engine.api.main import create_app
from rag_engine.config import Settings
from rag_engine.engine import Check, Engine
from rag_engine.generation.prompt import PromptBuilder
from rag_engine.ingestion.chunker import Chunker
from rag_engine.ingestion.pipeline import IngestionPipeline
from rag_engine.orchestrator import RagOrchestrator
from rag_engine.retrieval.retriever import Retriever
from rag_engine.schemas import SearchResponse
from rag_engine.store.pgvector import EmbeddingConfigMismatch


class FakeLLM:
    model = "fake-llm"

    def chat(self, messages, temperature=None) -> str:
        return "Có 12 ngày phép năm [1]."


def fake_engine(embedder, store) -> Engine:
    chunker = Chunker(min_section_tokens=0)
    retriever = Retriever(embedder, store, min_score=0.0)
    llm = FakeLLM()
    return Engine(
        settings=Settings(), embedder=embedder, store=store, chunker=chunker,
        pipeline=IngestionPipeline(chunker, embedder, store), retriever=retriever, llm=llm,
        orchestrator=RagOrchestrator(retriever, PromptBuilder(), llm),
    )


@pytest.fixture
def client(tmp_path, monkeypatch, fake_embedder, memory_store):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "a.md").write_text("# Nghỉ phép\nNhân viên có 12 ngày phép năm.\n",
                                            encoding="utf-8")
    (tmp_path / "docs" / "b.md").write_text("# Mật khẩu\nMật khẩu tối thiểu 12 ký tự.\n",
                                            encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    engine = fake_engine(fake_embedder, memory_store)
    app = create_app(
        Settings(),
        engine_factory=lambda s: engine,
        health_fn=lambda s: [Check("Postgres", True, "ok"), Check("LLM x", False, "missing", required=False)],
    )
    with TestClient(app) as c:
        c.engine = engine
        yield c


def test_health_reports_degraded_when_optional_missing(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "degraded"


def test_health_503_when_required_fails(fake_embedder, memory_store):
    app = create_app(Settings(), engine_factory=lambda s: fake_engine(fake_embedder, memory_store),
                     health_fn=lambda s: [Check("Postgres", False, "down")])
    with TestClient(app) as c:
        r = c.get("/health")
    assert r.status_code == 503 and r.json()["status"] == "error"


def test_ingest_search_ask_flow(client):
    r = client.post("/ingest", json={"path": "docs"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["added"] == 2 and body["chunks"] == 2
    assert {d["source"] for d in body["details"]} == {"docs/a.md", "docs/b.md"}

    assert client.get("/stats").json()["documents"] == 2

    r = client.post("/search", json={"query": "mật khẩu tối thiểu", "top_k": 1})
    assert r.status_code == 200
    assert r.json()["results"][0]["source"] == "docs/b.md"

    r = client.post("/ask", json={"question": "bao nhiêu ngày phép năm"})
    ans = r.json()
    assert r.status_code == 200 and ans["answered"] is True
    assert ans["cited"] == [1] and ans["sources"][0]["ref"] == 1
    assert {"embed", "search", "generate", "total"} <= set(ans["timings_ms"])


def test_search_matches_engine_and_cli_schema(client):
    client.post("/ingest", json={"path": "docs"})
    api = client.post("/search", json={"query": "nghỉ phép", "top_k": 2}).json()
    direct = SearchResponse.of(client.engine.retriever.retrieve("nghỉ phép", 2)).model_dump()
    strip = lambda d: [{k: v for k, v in r.items()} for r in d["results"]]  # noqa: E731
    assert strip(api) == strip(direct)


def test_validation_and_errors(client):
    assert client.post("/search", json={"query": ""}).status_code == 422
    assert client.post("/search", json={"query": "x", "top_k": 0}).status_code == 422
    assert client.post("/ingest", json={"path": "khong-ton-tai"}).status_code == 400


def test_ask_short_circuit_when_below_threshold(client):
    client.post("/ingest", json={"path": "docs"})
    ans = client.post("/ask", json={"question": "giá bitcoin", "min_score": 0.99}).json()
    assert ans["answered"] is False and ans["sources"] == []


def test_engine_config_mismatch_returns_409():
    def boom(_):
        raise EmbeddingConfigMismatch("lệch model")

    with TestClient(create_app(Settings(), engine_factory=boom, health_fn=lambda s: [])) as c:
        r = c.post("/search", json={"query": "x"})
    assert r.status_code == 409


def test_engine_unavailable_returns_503():
    def down(_):
        raise ConnectionError("db down")

    with TestClient(create_app(Settings(), engine_factory=down, health_fn=lambda s: [])) as c:
        assert c.get("/stats").status_code == 503
