from typing import Protocol


class LLMProvider(Protocol):
    model: str

    def chat(self, messages: list[dict], temperature: float | None = None) -> str:
        """messages theo dạng [{"role": "system"|"user"|"assistant", "content": str}]."""
        ...
