import hashlib
import re
from collections.abc import Iterator
from pathlib import Path

from rag_engine.models import Document

SUPPORTED_EXTENSIONS = {".md", ".markdown", ".txt"}
_H1_RE = re.compile(r"^#\s+(.+?)\s*#*\s*$", re.MULTILINE)


def to_source(path: Path, base: Path | None = None) -> str:
    """Định danh document: đường dẫn tương đối so với `base` (mặc định cwd), dạng posix."""
    base = (base or Path.cwd()).resolve()
    resolved = path.resolve()
    try:
        return resolved.relative_to(base).as_posix()
    except ValueError:
        return resolved.as_posix()


def iter_files(path: Path) -> Iterator[Path]:
    """Duyệt file được hỗ trợ (bỏ qua file/thư mục ẩn), theo thứ tự xác định."""
    if path.is_file():
        if path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield path
        return
    for p in sorted(path.rglob("*")):
        rel_parts = p.relative_to(path).parts
        if any(part.startswith(".") for part in rel_parts):
            continue
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield p


def load_file(path: Path, base: Path | None = None) -> Document:
    text = path.read_text(encoding="utf-8-sig").replace("\r\n", "\n")
    content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    title = None
    if path.suffix.lower() in {".md", ".markdown"} and (m := _H1_RE.search(text)):
        title = m.group(1)
    return Document(
        source=to_source(path, base),
        text=text,
        content_hash=content_hash,
        title=title or path.stem,
        metadata={"ext": path.suffix.lower(), "size_bytes": path.stat().st_size},
    )


def load_documents(path: str | Path, base: Path | None = None) -> list[Document]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Không tìm thấy đường dẫn: {p}")
    return [load_file(f, base) for f in iter_files(p)]
