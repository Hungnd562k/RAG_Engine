from rag_engine.generation.prompt import (
    NOT_FOUND_MESSAGE,
    REMINDER,
    PromptBuilder,
    extract_citations,
)
from rag_engine.ingestion.chunker import Chunker
from rag_engine.ingestion.pipeline import IngestionPipeline
from rag_engine.models import RetrievedChunk
from rag_engine.orchestrator import RagOrchestrator
from rag_engine.retrieval.retriever import Retriever


def rc(i: int, text: str, score: float = 0.8) -> RetrievedChunk:
    return RetrievedChunk(i, f"docs/{i}.md", text, ("H", f"S{i}"), score)


class FakeLLM:
    model = "fake-llm"

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.messages: list[dict] | None = None
        self.calls = 0

    def chat(self, messages, temperature=None) -> str:
        self.calls += 1
        self.messages = messages
        return self.reply


def test_prompt_numbers_context_in_order():
    msgs = PromptBuilder().build("Hỏi gì?", [rc(1, "alpha"), rc(2, "beta")])
    assert msgs[0]["role"] == "system" and "[n]" in msgs[0]["content"]
    user = msgs[1]["content"]
    assert user.index("[1] (source: docs/1.md › H › S1)\nalpha") < user.index("[2] (source: docs/2.md")
    assert "QUESTION: Hỏi gì?" in user
    assert user.rstrip().endswith(REMINDER)


def test_select_context_respects_budget_but_keeps_one():
    pb = PromptBuilder(context_max_chars=10)
    assert [c.chunk_id for c in pb.select_context([rc(1, "x" * 6), rc(2, "y" * 6)])] == [1]
    assert [c.chunk_id for c in pb.select_context([rc(1, "x" * 50)])] == [1]


def test_extract_citations_valid_invalid_and_grouped():
    valid, invalid = extract_citations("A [1]. B [2, 3]. C [1] [7].", n_sources=3)
    assert valid == [1, 2, 3]
    assert invalid == [7]
    assert extract_citations("không có", 3) == ([], [])


def make_orchestrator(tmp_path, monkeypatch, fake_embedder, memory_store, reply):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "a.md").write_text("# Nghỉ phép\nNhân viên có 12 ngày phép năm.\n",
                                            encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    IngestionPipeline(Chunker(min_section_tokens=0), fake_embedder, memory_store).ingest("docs")
    llm = FakeLLM(reply)
    retriever = Retriever(fake_embedder, memory_store, min_score=0.0)
    return RagOrchestrator(retriever, PromptBuilder(), llm), llm


def test_ask_returns_answer_with_citations(tmp_path, monkeypatch, fake_embedder, memory_store):
    orch, llm = make_orchestrator(tmp_path, monkeypatch, fake_embedder, memory_store,
                                  "Nhân viên có 12 ngày phép [1]. Xem thêm [5].")
    ans = orch.ask("bao nhiêu ngày phép năm", top_k=3)
    assert ans.answered and llm.calls == 1
    assert ans.sources[0].source == "docs/a.md"
    assert ans.cited == [1] and ans.invalid_citations == [5]
    assert {"embed", "search", "generate", "total"} <= set(ans.timings_ms)
    assert "12 ngày phép năm" in llm.messages[1]["content"]


def test_ask_short_circuits_without_context(tmp_path, monkeypatch, fake_embedder, memory_store):
    orch, llm = make_orchestrator(tmp_path, monkeypatch, fake_embedder, memory_store, "không gọi")
    ans = orch.ask("giá bitcoin", top_k=3, min_score=0.99)
    assert not ans.answered
    assert ans.text == NOT_FOUND_MESSAGE
    assert llm.calls == 0, "không có context đạt ngưỡng thì không gọi LLM"
    assert "generate" not in ans.timings_ms
