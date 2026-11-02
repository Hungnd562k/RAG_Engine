from dataclasses import dataclass, field


@dataclass(frozen=True)
class Document:
    source: str  # đường dẫn tương đối, khóa duy nhất
    text: str
    content_hash: str  # sha256 của text
    title: str | None = None
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Chunk:
    index: int
    text: str  # nội dung gốc, trích từ document.text[char_start:char_end]
    heading_path: tuple[str, ...]
    char_start: int
    char_end: int
    token_count: int

    def embed_text(self) -> str:
        """Text đưa vào embedding: thêm heading_path để vector mang ngữ cảnh tiêu đề."""
        if not self.heading_path:
            return self.text
        return " > ".join(self.heading_path) + "\n\n" + self.text


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: int
    source: str
    text: str
    heading_path: tuple[str, ...]
    score: float  # cosine similarity, 1 = giống nhất


@dataclass
class IngestReport:
    added: int = 0
    updated: int = 0
    skipped: int = 0
    deleted: int = 0
    chunks: int = 0
    elapsed_ms: float = 0.0
    details: list[tuple[str, str, int]] = field(default_factory=list)  # (source, action, chunks)


@dataclass
class SearchResult:
    results: list[RetrievedChunk]
    timings_ms: dict[str, float]


@dataclass
class Answer:
    text: str
    sources: list[RetrievedChunk]  # sources[i] tương ứng trích dẫn [i+1]
    cited: list[int]  # các số [n] hợp lệ xuất hiện trong câu trả lời
    invalid_citations: list[int]  # các số [n] không trỏ tới nguồn nào
    timings_ms: dict[str, float]
    answered: bool = True  # False khi short-circuit vì không có context đạt ngưỡng
