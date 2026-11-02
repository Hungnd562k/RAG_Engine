import httpx


class LLMError(RuntimeError):
    pass


class OllamaLLM:
    """Gọi Ollama `POST /api/chat` (stream=false) — ADR-0007."""

    def __init__(
        self,
        base_url: str,
        model: str = "qwen2.5:3b-instruct",
        temperature: float = 0.1,
        num_ctx: int = 4096,
        timeout_s: float = 300.0,
        client: httpx.Client | None = None,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.num_ctx = num_ctx
        self._client = client or httpx.Client(base_url=base_url, timeout=timeout_s)

    def chat(self, messages: list[dict], temperature: float | None = None) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.temperature if temperature is None else temperature,
                "num_ctx": self.num_ctx,
            },
        }
        try:
            resp = self._client.post("/api/chat", json=payload)
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise LLMError(
                f"Ollama trả lỗi {exc.response.status_code}: {exc.response.text[:200]}"
            ) from exc
        except httpx.TransportError as exc:
            raise LLMError(f"Không gọi được Ollama: {exc}") from exc
        return resp.json()["message"]["content"].strip()
