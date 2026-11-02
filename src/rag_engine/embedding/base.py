from typing import Protocol


class Embedder(Protocol):
    model: str
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Trả về một vector (đã chuẩn hóa L2) cho mỗi text, cùng thứ tự."""
        ...
