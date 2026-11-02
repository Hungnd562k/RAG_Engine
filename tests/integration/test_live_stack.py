"""Integration tests: cần `docker compose up -d` và model bge-m3 đã pull.

Dùng database riêng `rag_test` để không đụng dữ liệu dev.
"""

from pathlib import Path

import psycopg
import pytest
import yaml

from rag_engine.config import Settings
from rag_engine.engine import build_engine

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

    golden = yaml.safe_load((ROOT / "eval" / "golden.yaml").read_text(encoding="utf-8"))
    misses = []
    for item in golden:
        if item.get("out_of_scope"):
            continue
        res = engine.retriever.retrieve(item["question"], top_k=3, min_score=0.0)
        if not any(r.source in item["expected_sources"] for r in res.results):
            misses.append(item["id"])
    assert misses == [], f"Golden miss ở top-3: {misses}"


def test_ask_golden_answers_and_citations(engine, monkeypatch):
    """Cần model LLM (RAG_LLM_MODEL) đã pull. Chạy sau test ingest ở trên (cùng module)."""
    monkeypatch.chdir(ROOT)
    engine.pipeline.ingest("samples")
    golden = yaml.safe_load((ROOT / "eval" / "golden.yaml").read_text(encoding="utf-8"))
    failures = []
    for item in golden:
        ans = engine.orchestrator.ask(item["question"], top_k=5)
        if item.get("out_of_scope"):
            if ans.answered:
                failures.append((item["id"], "OOS không bị chặn", ans.text))
            continue
        missing = [s for s in item.get("answer_contains", []) if s not in ans.text]
        if missing or not ans.cited or ans.invalid_citations:
            failures.append((item["id"], f"missing={missing} cited={ans.cited} "
                                         f"invalid={ans.invalid_citations}", ans.text))
    assert failures == [], "\n".join(map(str, failures))


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
