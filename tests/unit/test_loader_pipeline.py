from rag_engine.ingestion.chunker import Chunker
from rag_engine.ingestion.loader import load_documents
from rag_engine.ingestion.pipeline import IngestionPipeline
from rag_engine.retrieval.retriever import Retriever


def write_corpus(root):
    (root / "docs").mkdir()
    (root / "docs" / "a.md").write_text("# Nghỉ phép\nNhân viên có 12 ngày phép năm.\n",
                                        encoding="utf-8")
    (root / "docs" / "b.txt").write_text("﻿Mật khẩu tối thiểu 12 ký tự.", encoding="utf-8")
    (root / "docs" / "skip.pdf").write_bytes(b"%PDF")
    (root / "docs" / ".hidden").mkdir()
    (root / "docs" / ".hidden" / "x.md").write_text("ẩn", encoding="utf-8")


def test_loader_filters_and_relative_sources(tmp_path):
    write_corpus(tmp_path)
    docs = load_documents(tmp_path / "docs", base=tmp_path)
    assert [d.source for d in docs] == ["docs/a.md", "docs/b.txt"]
    assert docs[0].title == "Nghỉ phép"
    assert not docs[1].text.startswith("﻿")


def make_pipeline(embedder, store):
    return IngestionPipeline(Chunker(min_section_tokens=0), embedder, store)


def test_ingest_is_idempotent_and_detects_changes(tmp_path, monkeypatch, fake_embedder,
                                                  memory_store):
    write_corpus(tmp_path)
    monkeypatch.chdir(tmp_path)
    pipe = make_pipeline(fake_embedder, memory_store)

    r1 = pipe.ingest("docs")
    assert (r1.added, r1.updated, r1.skipped) == (2, 0, 0)
    calls = fake_embedder.calls

    r2 = pipe.ingest("docs")
    assert (r2.added, r2.updated, r2.skipped) == (0, 0, 2)
    assert fake_embedder.calls == calls, "file không đổi thì không gọi embedding"

    (tmp_path / "docs" / "a.md").write_text("# Nghỉ phép\nNay là 14 ngày.\n", encoding="utf-8")
    r3 = pipe.ingest("docs")
    assert (r3.added, r3.updated, r3.skipped) == (0, 1, 1)


def test_prune_removes_missing_files_only_under_path(tmp_path, monkeypatch, fake_embedder,
                                                     memory_store):
    write_corpus(tmp_path)
    (tmp_path / "docs2").mkdir()
    (tmp_path / "docs2" / "c.md").write_text("khác", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    pipe = make_pipeline(fake_embedder, memory_store)
    pipe.ingest("docs")
    pipe.ingest("docs2")

    (tmp_path / "docs" / "b.txt").unlink()
    report = pipe.ingest("docs", prune=True)
    assert report.deleted == 1
    assert memory_store.list_sources() == ["docs/a.md", "docs2/c.md"]


def test_dry_run_writes_nothing(tmp_path, monkeypatch, fake_embedder, memory_store):
    write_corpus(tmp_path)
    monkeypatch.chdir(tmp_path)
    report = make_pipeline(fake_embedder, memory_store).ingest("docs", dry_run=True)
    assert report.added == 2
    assert memory_store.docs == {}
    assert fake_embedder.calls == 0


def test_retriever_ranks_and_filters(tmp_path, monkeypatch, fake_embedder, memory_store):
    write_corpus(tmp_path)
    monkeypatch.chdir(tmp_path)
    make_pipeline(fake_embedder, memory_store).ingest("docs")
    retriever = Retriever(fake_embedder, memory_store, min_score=0.0)

    res = retriever.retrieve("mật khẩu tối thiểu", top_k=2)
    assert res.results[0].source == "docs/b.txt"
    assert set(res.timings_ms) == {"embed", "search", "total"}

    assert retriever.retrieve("mật khẩu", top_k=2, min_score=0.99).results == []
    only_a = retriever.retrieve("mật khẩu", top_k=2, source_prefix="docs/a")
    assert {r.source for r in only_a.results} == {"docs/a.md"}
