import asyncio
from pathlib import Path

import pytest

from rag_engine.ingestion.chunker import Chunker
from rag_engine.ingestion.pipeline import IngestionPipeline
from rag_engine.mcp_server import EngineHolder, create_server, format_results, resolve_path
from rag_engine.models import RetrievedChunk, SearchResult
from rag_engine.retrieval.retriever import Retriever


class FakeEngine:
    def __init__(self, retriever, store):
        self.retriever, self.store = retriever, store
        self.closed = False

    def close(self):
        self.closed = True


def call(server, name, args) -> str:
    res = asyncio.run(server.call_tool(name, args))
    assert not res.is_error, res
    return "".join(c.text for c in res.content)


@pytest.fixture
def server(tmp_path, monkeypatch, fake_embedder, memory_store):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "a.md").write_text("# Nghỉ phép\nNhân viên có 12 ngày phép năm.\n",
                                            encoding="utf-8")
    (tmp_path / "docs" / "b.md").write_text("# VPN\nBắt buộc dùng VPN khi làm từ xa.\n",
                                            encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    IngestionPipeline(Chunker(min_section_tokens=0), fake_embedder, memory_store).ingest("docs")
    eng = FakeEngine(Retriever(fake_embedder, memory_store, min_score=0.0), memory_store)
    holder = EngineHolder(factory=lambda s: eng)
    return create_server(holder, tmp_path), eng


def test_resolve_path_relative_and_absolute(tmp_path):
    assert resolve_path("docs/a.md", tmp_path) == tmp_path / "docs" / "a.md"
    assert resolve_path("/abs/x.md", tmp_path) == Path("/abs/x.md")


def test_tools_are_listed(server):
    srv, _ = server
    names = {t.name for t in asyncio.run(srv.list_tools())}
    assert names == {"search_docs", "list_sources"}


def test_search_docs_returns_absolute_path_heading_and_text(server, tmp_path):
    srv, _ = server
    out = call(srv, "search_docs", {"query": "bao nhiêu ngày phép năm", "top_k": 1})
    assert out.startswith(f"[1] {tmp_path / 'docs' / 'a.md'} › Nghỉ phép")
    assert "12 ngày phép năm" in out and "score=" in out and "STALE" not in out


def test_search_docs_flags_stale_and_missing(server, tmp_path):
    srv, _ = server
    (tmp_path / "docs" / "a.md").write_text("# Nghỉ phép\nNay là 15 ngày.\n", encoding="utf-8")
    (tmp_path / "docs" / "b.md").unlink()
    out = call(srv, "search_docs", {"query": "ngày phép VPN", "top_k": 5})
    assert "[STALE" in out and "[MISSING" in out


def test_search_docs_source_prefix_and_top_k_clamp(server):
    srv, _ = server
    out = call(srv, "search_docs", {"query": "VPN", "top_k": 999, "source_prefix": "docs/b"})
    assert "docs/b.md" in out and "docs/a.md" not in out


def test_list_sources(server, tmp_path):
    srv, _ = server
    out = call(srv, "list_sources", {})
    assert out.startswith("2 file") and f"docs/a.md → {tmp_path / 'docs' / 'a.md'}" in out


def test_no_result_message_suggests_grep():
    res = SearchResult(results=[], timings_ms={}, top_score=0.41)
    msg = format_results(res, "kubernetes", Path("/"), lambda s: None, 0.5)
    assert "0.410" in msg and "grep" in msg


def test_engine_unavailable_is_tool_error_and_retried():
    calls = []

    def factory(s):
        calls.append(1)
        raise ConnectionError("connection refused")

    srv = create_server(EngineHolder(factory=factory), Path("/"))
    for _ in range(2):
        with pytest.raises(Exception, match="không sẵn sàng.*grep"):
            asyncio.run(srv.call_tool("search_docs", {"query": "x"}))
    assert len(calls) == 2, "lỗi hạ tầng không bị cache, lần gọi sau thử lại"


def test_rerank_score_shown():
    c = RetrievedChunk(1, "/x/a.md", "nội dung", ("H",), 0.61, rerank=0.93)
    out = format_results(SearchResult([c], {}, 0.61), "q", Path("/"), lambda s: None, 0.5)
    assert "score=0.610 rerank=0.930" in out and "[MISSING" in out
