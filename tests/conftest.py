import hashlib
import math
import re
from dataclasses import replace

import pytest

from rag_engine.models import Chunk, Document, RetrievedChunk


class FakeEmbedder:
    """Vector xác định theo bag-of-words băm, đủ để test luồng mà không cần Ollama."""

    model = "fake"
    dim = 64

    def __init__(self) -> None:
        self.calls = 0

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        out = []
        for t in texts:
            v = [0.0] * self.dim
            for w in t.lower().split():
                v[int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dim] += 1.0
            n = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / n for x in v])
        return out


class InMemoryStore:
    def __init__(self) -> None:
        self.docs: dict[str, tuple[Document, list[Chunk], list[list[float]]]] = {}

    def get_document_hash(self, source: str) -> str | None:
        return self.docs[source][0].content_hash if source in self.docs else None

    def replace_document(self, doc, chunks, vectors) -> None:
        self.docs[doc.source] = (doc, chunks, vectors)

    def delete_document(self, source: str) -> None:
        self.docs.pop(source, None)

    def list_sources(self, prefix: str = "") -> list[str]:
        return sorted(s for s in self.docs if s.startswith(prefix))

    def search(self, vector, top_k, source_prefix=None) -> list[RetrievedChunk]:
        hits = []
        for i, (doc, chunks, vecs) in enumerate(self.docs.values()):
            if source_prefix and not doc.source.startswith(source_prefix):
                continue
            for c, v in zip(chunks, vecs, strict=True):
                score = sum(a * b for a, b in zip(vector, v, strict=True))
                hits.append(RetrievedChunk(i * 1000 + c.index, doc.source, c.text,
                                           c.heading_path, score))
        return sorted(hits, key=lambda h: -h.score)[:top_k]

    def hybrid_search(self, query, vector, candidates, source_prefix=None) -> list[RetrievedChunk]:
        """Giả lập hybrid_search: coverage IDF trên tập từ (\\w và dấu chấm), không phân biệt hoa thường."""
        tok = lambda t: set(re.findall(r"[\w.]+", t.lower()))  # noqa: E731
        hits = self.search(vector, 10**9, source_prefix)
        all_hits = self.search(vector, 10**9)
        n = len(all_hits)
        terms = tok(query)
        idf = {t: math.log(1 + (n - df + 0.5) / (df + 0.5))
               for t in terms for df in [sum(t in tok(h.text) for h in all_hits)]}
        total = sum(idf.values()) or 1.0
        scored = [replace(h, lexical=sum(idf[t] for t in terms & tok(h.text)) / total) for h in hits]
        dense = scored[:candidates]
        lex = sorted((h for h in scored if h.lexical), key=lambda h: -h.lexical)[:candidates]
        return list({h.chunk_id: h for h in dense + lex}.values())

    def stats(self) -> dict:
        return {"documents": len(self.docs), "chunks": sum(len(c) for _, c, _ in self.docs.values())}

    def reset(self) -> None:
        self.docs.clear()

    def close(self) -> None:
        pass


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def memory_store() -> InMemoryStore:
    return InMemoryStore()
