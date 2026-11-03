from typing import Protocol


class Reranker(Protocol):
    def rerank(self, query: str, texts: list[str]) -> list[float]:
        """Điểm liên quan (cao = liên quan hơn) cho từng text, cùng thứ tự với `texts`."""
        ...
