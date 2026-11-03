import logging

import httpx

log = logging.getLogger(__name__)


class RerankError(RuntimeError):
    pass


class TEIReranker:
    """Cross-encoder qua HuggingFace Text Embeddings Inference `POST /rerank` (ADR-0011).

    TEI trả điểm đã qua sigmoid (0–1) và sắp xếp giảm dần kèm `index`; ở đây trả lại theo
    thứ tự đầu vào. Lỗi → RerankError (không âm thầm bỏ qua rerank, để số đo không bị sai).
    """

    def __init__(
        self,
        base_url: str,
        timeout_s: float = 60.0,
        batch_size: int = 32,
        client: httpx.Client | None = None,
    ) -> None:
        self.base_url = base_url
        self.batch_size = batch_size  # TEI mặc định --max-client-batch-size 32 (vượt → HTTP 413)
        self._client = client or httpx.Client(base_url=base_url, timeout=timeout_s)

    def rerank(self, query: str, texts: list[str]) -> list[float]:
        out: list[float] = []
        for i in range(0, len(texts), self.batch_size):
            out.extend(self._rerank_batch(query, texts[i : i + self.batch_size]))
        return out

    def _rerank_batch(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        payload = {"query": query, "texts": texts, "truncate": True}
        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                resp = self._client.post("/rerank", json=payload)
                resp.raise_for_status()
                ranked = resp.json()
                break
            except httpx.HTTPStatusError as exc:
                raise RerankError(
                    f"Reranker trả lỗi {exc.response.status_code}: {exc.response.text[:200]}"
                ) from exc
            except httpx.TransportError as exc:
                last_exc = exc
                log.warning("Reranker không phản hồi (lần %d): %s", attempt + 1, exc)
        else:
            raise RerankError(f"Không gọi được reranker {self.base_url}: {last_exc}") from last_exc

        scores: list[float | None] = [None] * len(texts)
        for r in ranked:
            scores[r["index"]] = float(r["score"])
        if any(s is None for s in scores):
            raise RerankError(f"Reranker trả {len(ranked)} điểm cho {len(texts)} text")
        return scores  # type: ignore[return-value]

    def info(self) -> dict:
        resp = self._client.get("/info")
        resp.raise_for_status()
        return resp.json()
