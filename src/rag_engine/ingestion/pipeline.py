import logging
import time
from pathlib import Path

from rag_engine.embedding.base import Embedder
from rag_engine.ingestion.chunker import Chunker
from rag_engine.ingestion.loader import load_documents, to_source
from rag_engine.models import IngestReport
from rag_engine.store.base import VectorStore

log = logging.getLogger(__name__)


class IngestionPipeline:
    """load → so hash → chunk → embed → upsert. Idempotent theo content_hash."""

    def __init__(self, chunker: Chunker, embedder: Embedder, store: VectorStore) -> None:
        self.chunker = chunker
        self.embedder = embedder
        self.store = store

    def ingest(self, path: str | Path, prune: bool = False, dry_run: bool = False) -> IngestReport:
        t0 = time.perf_counter()
        report = IngestReport()
        docs = load_documents(path)

        for doc in docs:
            existing = self.store.get_document_hash(doc.source)
            if existing == doc.content_hash:
                report.skipped += 1
                report.details.append((doc.source, "skipped", 0))
                continue

            action = "added" if existing is None else "updated"
            chunks = self.chunker.split(doc)
            if not dry_run:
                vectors = self.embedder.embed([c.embed_text() for c in chunks]) if chunks else []
                self.store.replace_document(doc, chunks, vectors)
            setattr(report, action, getattr(report, action) + 1)
            report.chunks += len(chunks)
            report.details.append((doc.source, action, len(chunks)))
            log.info("%s %s chunks=%d", action, doc.source, len(chunks))

        if prune:
            p = Path(path)
            prefix = to_source(p)
            if p.is_dir():
                prefix = "" if prefix == "." else prefix.rstrip("/") + "/"
            seen = {d.source for d in docs}
            for source in self.store.list_sources(prefix):
                if source not in seen:
                    if not dry_run:
                        self.store.delete_document(source)
                    report.deleted += 1
                    report.details.append((source, "deleted", 0))
                    log.info("deleted %s", source)

        report.elapsed_ms = (time.perf_counter() - t0) * 1000
        return report
