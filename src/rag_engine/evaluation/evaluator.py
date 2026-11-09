"""Đánh giá retrieval (và tùy chọn answer) trên golden set — Docs/06-evaluation-and-testing.md.

Metrics:
- hit@k: tỉ lệ câu trong phạm vi có ít nhất 1 chunk liên quan trong top-k.
- MRR: trung bình 1/rank của chunk liên quan đầu tiên (0 nếu không có trong top-k).
- in_scope_pass: tỉ lệ câu trong phạm vi có top_score ≥ min_score (không bị chặn nhầm).
  top_score = cosine dense cao nhất (SearchResult.top_score), kể cả ở chế độ hybrid.
- oos_block: tỉ lệ câu ngoài phạm vi bị chặn (top_score < min_score).
- Với --with-answer: answer_contains, tỉ lệ có trích dẫn hợp lệ, OOS trả answered=false.
"""

import math
import re
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml
from pydantic import BaseModel, model_validator

from rag_engine.models import RetrievedChunk
from rag_engine.orchestrator import RagOrchestrator
from rag_engine.retrieval.retriever import Retriever


class GoldenItem(BaseModel):
    id: str
    question: str
    expected_sources: list[str] = []
    expected_heading: str | None = None
    answer_contains: list[str] = []
    out_of_scope: bool = False

    @model_validator(mode="after")
    def _check(self) -> "GoldenItem":
        if not self.out_of_scope and not self.expected_sources:
            raise ValueError(f"{self.id}: câu trong phạm vi phải có expected_sources")
        return self


def load_golden(*paths: str | Path) -> list[GoldenItem]:
    items: list[GoldenItem] = []
    for p in paths:
        data = yaml.safe_load(Path(p).read_text(encoding="utf-8")) or []
        items.extend(GoldenItem(**d) for d in data)
    ids = [i.id for i in items]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise ValueError(f"Trùng id trong golden set: {sorted(dupes)}")
    return items


def source_matches(source: str, expected: str) -> bool:
    """Khớp chính xác, hoặc theo hậu tố đường dẫn (vd. 'E-Wallet/docs/prd.md' khớp đường dẫn tuyệt đối)."""
    return source == expected or source.endswith("/" + expected.lstrip("/"))


def is_relevant(chunk: RetrievedChunk, item: GoldenItem) -> bool:
    if not any(source_matches(chunk.source, e) for e in item.expected_sources):
        return False
    if item.expected_heading is None:
        return True
    want = item.expected_heading.casefold()
    return any(want in h.casefold() for h in chunk.heading_path)


_DIGIT_SEP_RE = re.compile(r"(?<=\d)[.,\s](?=\d{3}(?!\d))")


def normalize_answer(text: str) -> str:
    """Chuẩn hóa để chấm answer_contains: bỏ dấu phân cách hàng nghìn (50.000.000 = 50,000,000),
    không phân biệt hoa thường."""
    prev = None
    while prev != text:
        prev, text = text, _DIGIT_SEP_RE.sub("", text)
    return text.casefold()


def contains_all(answer: str, expected: list[str]) -> bool:
    norm = normalize_answer(answer)
    return all(normalize_answer(e) in norm for e in expected)


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = max(0, math.ceil(q / 100 * len(s)) - 1)
    return s[k]


@dataclass
class ItemResult:
    id: str
    question: str
    out_of_scope: bool
    rank: int | None  # thứ hạng chunk liên quan đầu tiên (1-based), None nếu không có trong top-k
    top1_score: float
    top1_source: str | None
    blocked: bool  # top_score < min_score → engine trả "không tìm thấy"
    embed_ms: float
    search_ms: float
    rerank_ms: float | None = None
    answer: str | None = None
    answered: bool | None = None
    answer_ok: bool | None = None  # chứa đủ answer_contains
    citation_ok: bool | None = None  # có ≥1 trích dẫn hợp lệ, không có trích dẫn sai
    unsupported_numbers: list[str] | None = None  # con số không có trong context (R11)
    generate_ms: float | None = None


@dataclass
class EvalReport:
    n_in_scope: int
    n_oos: int
    top_k: int
    min_score: float
    metrics: dict[str, float]
    items: list[ItemResult] = field(default_factory=list)

    def misses(self, k: int) -> list[ItemResult]:
        return [r for r in self.items if not r.out_of_scope and (r.rank is None or r.rank > k)]

    def to_dict(self) -> dict:
        return {
            "n_in_scope": self.n_in_scope,
            "n_oos": self.n_oos,
            "top_k": self.top_k,
            "min_score": self.min_score,
            "metrics": self.metrics,
            "items": [asdict(r) for r in self.items],
        }


def evaluate(
    items: list[GoldenItem],
    retriever: Retriever,
    orchestrator: RagOrchestrator | None = None,
    top_k: int = 5,
    ks: tuple[int, ...] = (1, 3, 5),
) -> EvalReport:
    depth = max(top_k, *ks)
    threshold = retriever.min_score
    results: list[ItemResult] = []

    for it in items:
        res = retriever.retrieve(it.question, depth, min_score=-1.0)
        hits = res.results
        rank = next((i for i, h in enumerate(hits, 1) if is_relevant(h, it)), None)
        top1 = res.top_score if res.top_score is not None else -1.0
        r = ItemResult(
            id=it.id,
            question=it.question,
            out_of_scope=it.out_of_scope,
            rank=rank,
            top1_score=top1,
            top1_source=hits[0].source if hits else None,
            blocked=top1 < threshold,
            embed_ms=res.timings_ms["embed"],
            search_ms=res.timings_ms["search"],
            rerank_ms=res.timings_ms.get("rerank"),
        )
        if orchestrator is not None:
            ans = orchestrator.ask(it.question, top_k)
            r.answer, r.answered = ans.text, ans.answered
            r.generate_ms = ans.timings_ms.get("generate")
            r.unsupported_numbers = ans.unsupported_numbers if ans.answered else None
            if it.out_of_scope:
                r.answer_ok = not ans.answered
                r.citation_ok = None
            else:
                r.answer_ok = ans.answered and contains_all(ans.text, it.answer_contains)
                r.citation_ok = bool(ans.cited) and not ans.invalid_citations
        results.append(r)

    ins = [r for r in results if not r.out_of_scope]
    oos = [r for r in results if r.out_of_scope]
    m: dict[str, float] = {}
    for k in ks:
        m[f"hit@{k}"] = _ratio(sum(1 for r in ins if r.rank is not None and r.rank <= k), len(ins))
    m["mrr"] = _ratio(sum(1 / r.rank for r in ins if r.rank is not None and r.rank <= top_k), len(ins))
    m["in_scope_pass"] = _ratio(sum(1 for r in ins if not r.blocked), len(ins))
    m["oos_block"] = _ratio(sum(1 for r in oos if r.blocked), len(oos))
    if ins:
        m["min_in_scope_top1"] = min(r.top1_score for r in ins)
    if oos:
        m["max_oos_top1"] = max(r.top1_score for r in oos)
    embeds = [r.embed_ms for r in results]
    searches = [r.search_ms for r in results]
    m.update(embed_p50_ms=statistics.median(embeds) if embeds else 0.0,
             embed_p95_ms=percentile(embeds, 95),
             search_p50_ms=statistics.median(searches) if searches else 0.0,
             search_p95_ms=percentile(searches, 95))
    reranks = [r.rerank_ms for r in results if r.rerank_ms is not None]
    if reranks:
        m["rerank_p50_ms"] = statistics.median(reranks)
        m["rerank_p95_ms"] = percentile(reranks, 95)
    if orchestrator is not None:
        m["answer_ok"] = _ratio(sum(1 for r in ins if r.answer_ok), len(ins))
        m["citation_ok"] = _ratio(sum(1 for r in ins if r.citation_ok), len(ins))
        # Tỉ lệ câu trả lời (in-scope, đã trả lời) không có con số nào ngoài context.
        answered = [r for r in ins if r.answered]
        m["numbers_grounded"] = _ratio(sum(1 for r in answered if not r.unsupported_numbers),
                                       len(answered))
        m["oos_not_answered"] = _ratio(sum(1 for r in oos if r.answer_ok), len(oos))
        gens = [r.generate_ms for r in results if r.generate_ms is not None]
        if gens:
            m["generate_p50_ms"] = statistics.median(gens)
            m["generate_p95_ms"] = percentile(gens, 95)

    return EvalReport(len(ins), len(oos), top_k, threshold, m, results)


def _ratio(num: float, den: int) -> float:
    return num / den if den else float("nan")
