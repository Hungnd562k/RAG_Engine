import logging
import math

import httpx

log = logging.getLogger(__name__)


class EmbeddingError(RuntimeError):
    pass


class OllamaEmbedder:
    """Gọi Ollama `POST /api/embed` theo batch (ADR-0004)."""

    def __init__(
        self,
        base_url: str,
        model: str = "bge-m3",
        dim: int = 1024,
        batch_size: int = 16,
        timeout_s: float = 300.0,
        num_gpu: int | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        self.model = model
        self.dim = dim
        self.batch_size = batch_size
        self.num_gpu = num_gpu
        self._client = client or httpx.Client(base_url=base_url, timeout=timeout_s)

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        payload: dict = {"model": self.model, "input": texts, "truncate": True}
        if self.num_gpu is not None:
            payload["options"] = {"num_gpu": self.num_gpu}
        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                resp = self._client.post("/api/embed", json=payload)
                resp.raise_for_status()
                vectors = resp.json()["embeddings"]
                break
            except httpx.HTTPStatusError as exc:
                raise EmbeddingError(f"Ollama trả lỗi {exc.response.status_code}: "
                                     f"{exc.response.text[:200]}") from exc
            except httpx.TransportError as exc:
                last_exc = exc
                log.warning("Ollama không phản hồi (lần %d): %s", attempt + 1, exc)
        else:
            raise EmbeddingError(f"Không gọi được Ollama: {last_exc}") from last_exc

        if len(vectors) != len(texts):
            raise EmbeddingError(f"Số vector ({len(vectors)}) khác số text ({len(texts)})")
        out = []
        for v in vectors:
            if len(v) != self.dim:
                raise EmbeddingError(
                    f"Model {self.model} trả vector {len(v)} chiều, cấu hình là {self.dim}"
                )
            norm = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / norm for x in v])
        return out

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            out.extend(self._embed_batch(texts[i : i + self.batch_size]))
        return out

    def list_models(self) -> list[str]:
        resp = self._client.get("/api/tags")
        resp.raise_for_status()
        return [m["name"] for m in resp.json().get("models", [])]
