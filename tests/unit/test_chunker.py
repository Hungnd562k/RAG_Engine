import hashlib

import pytest

from rag_engine.ingestion.chunker import Chunker, split_sections
from rag_engine.models import Document


def make_doc(text: str, ext: str = ".md") -> Document:
    return Document("t" + ext, text, hashlib.sha256(text.encode()).hexdigest(), metadata={"ext": ext})


def test_empty_document_has_no_chunks():
    assert Chunker().split(make_doc("   \n\n ")) == []


def test_short_text_is_single_chunk():
    chunks = Chunker().split(make_doc("Xin chào thế giới.", ".txt"))
    assert len(chunks) == 1
    assert chunks[0].text == "Xin chào thế giới."
    assert chunks[0].heading_path == ()


def test_heading_path_tracks_nested_headings():
    text = "# A\nintro a\n## B\nnội dung b\n### C\nnội dung c\n## D\nnội dung d\n"
    paths = [s.heading_path for s in split_sections(text)]
    assert paths == [("A",), ("A", "B"), ("A", "B", "C"), ("A", "D")]


def test_plain_text_splits_on_paragraphs():
    paras = [f"Chủ đề {i}: " + "nội dung " * 40 for i in range(3)]
    text = "\n\n".join(paras)
    chunks = Chunker(min_section_tokens=10).split(make_doc(text, ".txt"))
    assert [c.text for c in chunks] == [p.strip() for p in paras]


def test_headings_inside_code_fence_are_ignored():
    text = "# A\n```\n# not a heading\n```\nbody\n"
    assert [s.heading_path for s in split_sections(text)] == [("A",)]


def test_small_sections_merge_into_next():
    text = "# Parent\nngắn\n## Child\n" + "nội dung dài " * 50
    chunks = Chunker(min_section_tokens=10).split(make_doc(text))
    assert len(chunks) == 1
    assert chunks[0].heading_path == ("Parent", "Child")
    assert chunks[0].text.startswith("# Parent")


@pytest.mark.parametrize("ext", [".md", ".txt"])
def test_offsets_match_source_and_size_bounded(ext):
    para = "Câu số {i} nói về chính sách nghỉ phép năm của công ty. "
    text = "# Title\n\n" + "\n\n".join("".join(para.format(i=i * 10 + j) for j in range(5))
                                      for i in range(20))
    chunker = Chunker(chunk_size=64, chunk_overlap=16, min_section_tokens=0)
    doc = make_doc(text, ext)
    chunks = chunker.split(doc)
    assert len(chunks) > 3
    for c in chunks:
        assert doc.text[c.char_start : c.char_end] == c.text
        assert len(c.text) <= chunker.max_chars
        assert c.text == c.text.strip()
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_consecutive_chunks_overlap():
    text = " ".join(f"tu{i}." for i in range(400))
    chunks = Chunker(chunk_size=50, chunk_overlap=10, min_section_tokens=0).split(
        make_doc(text, ".txt")
    )
    assert len(chunks) > 2
    for a, b in zip(chunks, chunks[1:], strict=False):
        assert b.char_start < a.char_end, "chunk sau phải bắt đầu trước khi chunk trước kết thúc"
        assert b.char_start > a.char_start


def test_no_word_split_when_spaces_available():
    text = " ".join(["nghiêng"] * 300)
    for c in Chunker(chunk_size=20, chunk_overlap=4).split(make_doc(text, ".txt")):
        assert set(c.text.split()) == {"nghiêng"}


def test_overlap_must_be_smaller_than_size():
    with pytest.raises(ValueError):
        Chunker(chunk_size=10, chunk_overlap=10)


def test_embed_text_prefixes_heading_path():
    text = "# Nghỉ phép\n## Năm\n" + "12 ngày phép. " * 30
    c = Chunker(min_section_tokens=0).split(make_doc(text))[-1]
    assert c.embed_text().startswith("Nghỉ phép > Năm\n\n")
