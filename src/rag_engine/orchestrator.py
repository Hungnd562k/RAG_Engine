import time

from rag_engine.generation.grounding import unsupported_numbers
from rag_engine.generation.prompt import NOT_FOUND_MESSAGE, PromptBuilder, extract_citations
from rag_engine.llm.base import LLMProvider
from rag_engine.models import Answer
from rag_engine.retrieval.retriever import Retriever


class RagOrchestrator:
    """ask(question) → retrieve → prompt → LLM → Answer có trích dẫn và timings."""

    def __init__(self, retriever: Retriever, prompt_builder: PromptBuilder, llm: LLMProvider) -> None:
        self.retriever = retriever
        self.prompt_builder = prompt_builder
        self.llm = llm

    def ask(
        self,
        question: str,
        top_k: int,
        source_prefix: str | None = None,
        min_score: float | None = None,
    ) -> Answer:
        t0 = time.perf_counter()
        retrieved = self.retriever.retrieve(
            question, top_k, source_prefix=source_prefix, min_score=min_score
        )
        timings = {k: v for k, v in retrieved.timings_ms.items() if k != "total"}

        if not retrieved.results:
            timings["total"] = (time.perf_counter() - t0) * 1000
            return Answer(NOT_FOUND_MESSAGE, [], [], [], timings, answered=False)

        sources = self.prompt_builder.select_context(retrieved.results)
        messages = self.prompt_builder.build(question, sources)
        t1 = time.perf_counter()
        text = self.llm.chat(messages)
        timings["generate"] = (time.perf_counter() - t1) * 1000
        timings["total"] = (time.perf_counter() - t0) * 1000

        cited, invalid = extract_citations(text, len(sources))
        return Answer(text, sources, cited, invalid, timings, answered=True,
                      unsupported_numbers=unsupported_numbers(text, sources, question))
