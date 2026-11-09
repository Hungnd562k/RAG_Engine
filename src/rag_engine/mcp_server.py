"""MCP server (stdio) cho agent (Claude) tra cứu tài liệu đã ingest: `rag mcp`.

Chỉ đọc: `search_docs` và `list_sources`. Không có `ask`: agent tự đọc chunk và tự trả lời, nên
không cần LLM 3B. Không có ingest/reset: thay đổi index là việc của người dùng (`rag ingest`).

stdout là kênh JSON-RPC: module này không được print ra stdout.
"""

import hashlib
import threading
from pathlib import Path

from rag_engine.config import Settings
from rag_engine.engine import Engine, build_engine
from rag_engine.generation.prompt import format_source
from rag_engine.models import SearchResult

INSTRUCTIONS = """\
Tra cứu ngữ nghĩa (semantic search) trên tài liệu dự án đã được ingest vào RAG engine local.

Dùng `search_docs` khi cần tìm thông tin theo Ý NGHĨA trong tài liệu (quy tắc nghiệp vụ, kiến trúc,
quy trình, quyết định thiết kế), khi không biết file nào chứa câu trả lời, hoặc khi câu hỏi khác
ngôn ngữ với tài liệu (hỏi tiếng Anh, tài liệu tiếng Việt và ngược lại).
Dùng grep thay vì search_docs khi cần chuỗi chính xác (mã lỗi, tên biến, định danh) hoặc cần tìm
ĐỦ mọi chỗ xuất hiện; search_docs chỉ trả top-k.

Kết quả là các đoạn (chunk) kèm đường dẫn file tuyệt đối và heading. Chunk có thể thiếu ngữ cảnh:
khi cần chi tiết, Read file nguồn. Chunk đánh dấu STALE nghĩa là file đã đổi sau lần ingest:
Read file để có nội dung hiện tại. Không có kết quả nghĩa là tài liệu không nói về chủ đề đó
(hoặc chưa được ingest): thử lại bằng từ khóa khác, rồi chuyển sang grep."""

MAX_TOP_K = 20


class EngineHolder:
    """Khởi tạo engine lần đầu cần dùng; lỗi hạ tầng không làm chết server, lần gọi sau thử lại."""

    def __init__(self, settings: Settings | None = None, factory=build_engine) -> None:
        self._settings = settings
        self._factory = factory
        self._engine: Engine | None = None
        self._lock = threading.Lock()

    def get(self) -> Engine:
        with self._lock:
            if self._engine is None:
                self._engine = self._factory(self._settings or Settings())
            return self._engine

    def close(self) -> None:
        with self._lock:
            if self._engine is not None:
                self._engine.close()
                self._engine = None


def resolve_path(source: str, docs_root: Path) -> Path:
    """source lưu tương đối so với thư mục chạy ingest, hoặc tuyệt đối nếu nằm ngoài thư mục đó."""
    p = Path(source)
    return p if p.is_absolute() else (docs_root / p).resolve()


def file_hash(path: Path) -> str | None:
    """Cùng cách băm với loader (utf-8-sig, chuẩn hóa CRLF); None nếu file không đọc được."""
    try:
        text = path.read_text(encoding="utf-8-sig").replace("\r\n", "\n")
    except (OSError, UnicodeDecodeError):
        return None
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def format_results(res: SearchResult, query: str, docs_root: Path, stored_hash, min_score: float) -> str:
    """Render kết quả cho agent: đường dẫn tuyệt đối, heading, điểm, cờ STALE/MISSING, nội dung chunk."""
    if not res.results:
        best = f"{res.top_score:.3f}" if res.top_score is not None else "n/a"
        return (
            f'Không có đoạn nào liên quan tới "{query}" (cosine cao nhất {best} < ngưỡng {min_score}). '
            "Tài liệu có thể không nói về chủ đề này. Thử từ khóa khác hoặc dùng grep."
        )
    status: dict[str, str] = {}
    blocks = []
    for i, c in enumerate(res.results, 1):
        path = resolve_path(c.source, docs_root)
        if c.source not in status:
            h = file_hash(path)
            status[c.source] = (
                " [MISSING: file không còn tồn tại]" if h is None
                else " [STALE: file đã đổi sau lần ingest, Read file để có bản mới]"
                if h != stored_hash(c.source) else ""
            )
        scores = f"score={c.score:.3f}" + (f" rerank={c.rerank:.3f}" if c.rerank is not None else "")
        heading = format_source(c)[len(c.source):]  # " › H1 › H2" hoặc ""
        blocks.append(f"[{i}] {path}{heading}\n({scores}){status[c.source]}\n{c.text.strip()}")
    return "\n\n---\n\n".join(blocks)


def create_server(holder: EngineHolder | None = None, docs_root: Path | None = None):
    from mcp.server.mcpserver import MCPServer
    from mcp.server.mcpserver.exceptions import ToolError

    holder = holder or EngineHolder()
    root = (docs_root or Path.cwd()).resolve()
    server = MCPServer(name="rag-docs", instructions=INSTRUCTIONS)

    def engine() -> Engine:
        try:
            return holder.get()
        except Exception as exc:  # hạ tầng chưa chạy: báo rõ để agent chuyển sang grep
            raise ToolError(
                f"RAG engine không sẵn sàng ({exc}). Dùng grep/Read thay thế; người dùng cần chạy "
                "`docker compose up -d` trong RAG_Engine."
            ) from exc

    @server.tool()
    def search_docs(query: str, top_k: int = 5, source_prefix: str | None = None) -> str:
        """Tìm các đoạn tài liệu liên quan nhất tới `query` theo ngữ nghĩa (hybrid hoặc rerank).

        Args:
            query: câu hỏi hoặc mô tả thông tin cần tìm, tiếng Việt hoặc tiếng Anh.
            top_k: số đoạn trả về (1-20, mặc định 5).
            source_prefix: chỉ tìm trong các source bắt đầu bằng tiền tố này
                (xem `list_sources`), ví dụ "/home/user/project/docs" hoặc "samples/".
        """
        if not query.strip():
            raise ToolError("query rỗng")
        eng = engine()
        k = max(1, min(top_k, MAX_TOP_K))
        try:
            res = eng.retriever.retrieve(query, k, source_prefix=source_prefix or None)
        except Exception as exc:
            raise ToolError(f"Tìm kiếm lỗi: {exc}. Dùng grep/Read thay thế.") from exc
        return format_results(res, query, root, eng.store.get_document_hash, eng.retriever.min_score)

    @server.tool()
    def list_sources(prefix: str = "") -> str:
        """Liệt kê các file tài liệu đã được ingest (đường dẫn tuyệt đối), lọc theo tiền tố source."""
        eng = engine()
        sources = eng.store.list_sources(prefix)
        if not sources:
            return f'Không có source nào bắt đầu bằng "{prefix}".'
        lines = [f"{len(sources)} file đã ingest (source → đường dẫn):"]
        for s in sources:
            p = resolve_path(s, root)
            lines.append(s if str(p) == s else f"{s} → {p}")
        return "\n".join(lines)

    return server
