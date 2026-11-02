"""Chunking: tách theo heading Markdown, sau đó recursive split theo separator.

Mọi chunk là một đoạn liên tục của text gốc: `chunk.text == doc.text[char_start:char_end]`.
Kích thước tính bằng token ước lượng = số ký tự / `chars_per_token` (xem ADR-0006).
"""

import re
from dataclasses import dataclass

from rag_engine.models import Chunk, Document

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_FENCE_RE = re.compile(r"^\s*(```|~~~)")
DEFAULT_SEPARATORS = ("\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ")

Span = tuple[int, int]


@dataclass(frozen=True)
class Section:
    heading_path: tuple[str, ...]
    start: int
    end: int


def split_sections(text: str, is_markdown: bool = True) -> list[Section]:
    """Tách text thành các section liên tục, mỗi section bắt đầu tại một dòng heading.

    Bỏ qua heading nằm trong code fence. Text trước heading đầu tiên là section không heading.
    Text thuần (không có heading): mỗi đoạn văn (ngăn bởi dòng trống) là một section, để
    một file ngắn nhiều chủ đề không bị dồn thành một chunk "loãng".
    """
    if not is_markdown:
        return [Section((), s, e) for s, e in _pieces(text, 0, len(text), "\n\n") if text[s:e].strip()]

    sections: list[Section] = []
    stack: list[tuple[int, str]] = []  # (level, title)
    current_path: tuple[str, ...] = ()
    current_start = 0
    in_fence = False
    pos = 0
    for line in text.splitlines(keepends=True):
        stripped = line.rstrip("\n")
        if _FENCE_RE.match(stripped):
            in_fence = not in_fence
        elif not in_fence and (m := _HEADING_RE.match(stripped)):
            if pos > current_start:
                sections.append(Section(current_path, current_start, pos))
            level, title = len(m.group(1)), m.group(2)
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            current_path = tuple(t for _, t in stack)
            current_start = pos
        pos += len(line)
    if pos > current_start:
        sections.append(Section(current_path, current_start, pos))
    return [s for s in sections if text[s.start : s.end].strip()]


def _pieces(text: str, start: int, end: int, sep: str) -> list[Span]:
    """Cắt [start, end) theo sep, giữ sep ở cuối mỗi mảnh để các mảnh liền nhau."""
    out: list[Span] = []
    i = start
    while (j := text.find(sep, i, end)) != -1:
        out.append((i, j + len(sep)))
        i = j + len(sep)
    if i < end:
        out.append((i, end))
    return out


def _atomic_spans(
    text: str, start: int, end: int, seps: tuple[str, ...], max_chars: int
) -> list[Span]:
    if end - start <= max_chars:
        return [(start, end)]
    for k, sep in enumerate(seps):
        pieces = _pieces(text, start, end, sep)
        if len(pieces) > 1:
            out: list[Span] = []
            for s, e in pieces:
                out.extend(_atomic_spans(text, s, e, seps[k + 1 :], max_chars))
            return out
    # Không còn separator: cắt cứng.
    return [(s, min(s + max_chars, end)) for s in range(start, end, max_chars)]


def _merge_spans(atoms: list[Span], max_chars: int, overlap_chars: int) -> list[Span]:
    """Gộp các span nguyên tử thành chunk <= max_chars, chunk sau lặp lại đuôi chunk trước."""
    chunks: list[Span] = []
    cur: list[Span] = []
    for atom in atoms:
        if cur and atom[1] - cur[0][0] > max_chars:
            chunks.append((cur[0][0], cur[-1][1]))
            tail: list[Span] = []
            for s in reversed(cur):
                if cur[-1][1] - s[0] > overlap_chars:
                    break
                tail.insert(0, s)
            cur = tail if tail and atom[1] - tail[0][0] <= max_chars else []
        cur.append(atom)
    if cur:
        chunks.append((cur[0][0], cur[-1][1]))
    return chunks


def _strip_span(text: str, start: int, end: int) -> Span:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


class Chunker:
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        chars_per_token: float = 4.0,
        min_section_tokens: int = 50,
        separators: tuple[str, ...] = DEFAULT_SEPARATORS,
    ) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap phải nhỏ hơn chunk_size")
        self.chars_per_token = chars_per_token
        self.max_chars = int(chunk_size * chars_per_token)
        self.overlap_chars = int(chunk_overlap * chars_per_token)
        self.min_section_chars = int(min_section_tokens * chars_per_token)
        self.separators = separators

    def estimate_tokens(self, text: str) -> int:
        return max(1, round(len(text) / self.chars_per_token))

    def _merge_small_sections(self, text: str, sections: list[Section]) -> list[Section]:
        """Section quá nhỏ (vd. heading cha + 1 câu dẫn) được gộp vào section ngay sau nó."""
        out: list[Section] = []
        pending: Section | None = None
        for sec in sections:
            if pending is not None:
                sec = Section(sec.heading_path, pending.start, sec.end)
                pending = None
            if len(text[sec.start : sec.end].strip()) < self.min_section_chars:
                pending = sec
            else:
                out.append(sec)
        if pending is not None:
            if out and pending.start == out[-1].end:
                last = out.pop()
                out.append(Section(last.heading_path, last.start, pending.end))
            else:
                out.append(pending)
        return out

    def split(self, doc: Document) -> list[Chunk]:
        text = doc.text
        is_md = doc.metadata.get("ext", ".md") in {".md", ".markdown"}
        sections = self._merge_small_sections(text, split_sections(text, is_md))
        chunks: list[Chunk] = []
        for sec in sections:
            atoms = _atomic_spans(text, sec.start, sec.end, self.separators, self.max_chars)
            for s, e in _merge_spans(atoms, self.max_chars, self.overlap_chars):
                s, e = _strip_span(text, s, e)
                if s >= e:
                    continue
                piece = text[s:e]
                chunks.append(
                    Chunk(
                        index=len(chunks),
                        text=piece,
                        heading_path=sec.heading_path,
                        char_start=s,
                        char_end=e,
                        token_count=self.estimate_tokens(piece),
                    )
                )
        return chunks
