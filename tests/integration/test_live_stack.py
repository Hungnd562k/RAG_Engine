"""Integration tests: cần `docker compose up -d` và model bge-m3 đã pull.

Dùng database riêng `rag_test` để không đụng dữ liệu dev.
"""

from pathlib import Path

import psycopg
import pytest

from rag_engine.config import Settings
from rag_engine.engine import build_engine
from rag_engine.evaluation.evaluator import evaluate, load_golden

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def engine():
    base = Settings()
    admin_url = base.database_url
    test_url = admin_url.rsplit("/", 1)[0] + "/rag_test"
    try:
        with psycopg.connect(admin_url, autocommit=True, connect_timeout=3) as c:
            c.execute("DROP DATABASE IF EXISTS rag_test WITH (FORCE)")
            c.execute("CREATE DATABASE rag_test")
    except psycopg.OperationalError as exc:
        pytest.skip(f"Postgres không sẵn sàng: {exc}")
    eng = build_engine(base.model_copy(update={"database_url": test_url}))
    yield eng
    eng.close()
    with psycopg.connect(admin_url, autocommit=True) as c:
        c.execute("DROP DATABASE IF EXISTS rag_test WITH (FORCE)")


def cosine(a, b):
    return sum(x * y for x, y in zip(a, b, strict=True))


def test_embedder_dim_and_semantics(engine):
    v = engine.embedder.embed(["nghỉ phép năm", "annual leave", "giá vàng hôm nay"])
    assert all(len(x) == 1024 for x in v)
    assert cosine(v[0], v[1]) > cosine(v[0], v[2]), "cross-lingual: VI gần EN đồng nghĩa hơn"


def test_ingest_idempotent_and_golden_hit_at_3(engine, monkeypatch):
    monkeypatch.chdir(ROOT)
    r1 = engine.pipeline.ingest("samples")
    assert r1.added > 0 and r1.chunks > 0
    r2 = engine.pipeline.ingest("samples")
    assert r2.added == r2.updated == 0 and r2.skipped == r1.added

    stats = engine.store.stats()
    assert stats["documents"] == r1.added and stats["chunks"] == r1.chunks

    # Retrieval trên samples (DB chỉ có samples): ngưỡng theo baseline M5 (Docs/06).
    rep = evaluate(load_golden(ROOT / "eval" / "golden.yaml"), engine.retriever, top_k=5)
    m = rep.metrics
    detail = [(r.id, r.rank, round(r.top1_score, 3)) for r in rep.misses(3)]
    assert m["hit@3"] == 1.0, f"Golden miss ở top-3: {detail}"
    assert m["hit@1"] >= 0.9, f"hit@1={m['hit@1']:.3f}: {[(r.id, r.rank) for r in rep.misses(1)]}"
    assert m["in_scope_pass"] == 1.0 and m["oos_block"] == 1.0, m


def test_ask_golden_answers_and_citations(engine, monkeypatch):
    """Cần model LLM (RAG_LLM_MODEL) đã pull. LLM không tất định → dùng ngưỡng, không đòi 100%."""
    monkeypatch.chdir(ROOT)
    engine.pipeline.ingest("samples")
    rep = evaluate(load_golden(ROOT / "eval" / "golden.yaml"), engine.retriever,
                   engine.orchestrator, top_k=5)
    m = rep.metrics
    bad = [(r.id, r.answer) for r in rep.items if r.answer_ok is False or r.citation_ok is False]
    assert m["answer_ok"] >= 0.85 and m["citation_ok"] >= 0.85, (m, bad)
    assert m["oos_not_answered"] == 1.0, bad


def test_api_against_live_engine(engine, monkeypatch):
    """M4 DoD: /search trùng với engine (cùng schema với `rag search --json`), /ask có trích dẫn."""
    from fastapi.testclient import TestClient

    from rag_engine.api.main import create_app
    from rag_engine.schemas import SearchResponse

    class NoClose:  # engine của fixture dùng chung cho cả module, lifespan không được đóng nó
        def __init__(self, e):
            self._e = e

        def __getattr__(self, name):
            return getattr(self._e, name)

        def close(self):
            pass

    monkeypatch.chdir(ROOT)
    app = create_app(engine.settings, engine_factory=lambda s: NoClose(engine))
    with TestClient(app) as c:
        assert c.post("/ingest", json={"path": "samples"}).status_code == 200
        api = c.post("/search", json={"query": "chính sách nghỉ phép năm", "top_k": 3}).json()
        direct = SearchResponse.of(engine.retriever.retrieve("chính sách nghỉ phép năm", 3)).model_dump()
        key = lambda d: [(r["chunk_id"], r["source"]) for r in d["results"]]  # noqa: E731
        assert key(api) == key(direct)

        ans = c.post("/ask", json={"question": "Mật khẩu phải dài tối thiểu bao nhiêu ký tự?"}).json()
        assert ans["answered"] and "12" in ans["answer"] and ans["cited"]
        assert c.post("/ask", json={"question": "Giá bitcoin hôm nay?"}).json()["answered"] is False
