import logging
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from rag_engine.config import get_settings
from rag_engine.embedding.ollama import EmbeddingError
from rag_engine.engine import Engine, build_engine, build_store, health_checks
from rag_engine.generation.prompt import format_source
from rag_engine.llm.ollama import LLMError
from rag_engine.schemas import AskResponse, SearchResponse
from rag_engine.store.pgvector import EmbeddingConfigMismatch

app = typer.Typer(help="RAG engine CLI (xem Docs/04-interfaces-cli-api.md)", no_args_is_help=True)
console = Console()


@app.callback()
def _setup(verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False) -> None:
    level = "DEBUG" if verbose else get_settings().log_level
    logging.basicConfig(
        level=level, format="%(message)s", handlers=[RichHandler(console=console, show_path=False)]
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)


def _engine() -> Engine:
    try:
        return build_engine(get_settings())
    except EmbeddingConfigMismatch as exc:
        console.print(f"[red]✗[/] {exc}")
        raise typer.Exit(2) from exc
    except Exception as exc:
        console.print(f"[red]✗ Không khởi tạo được engine:[/] {exc}\nChạy `rag health` để kiểm tra.")
        raise typer.Exit(1) from exc


def _fmt_ms(timings: dict[str, float]) -> str:
    return " ".join(f"{k}={v:.0f}ms" for k, v in timings.items())


@app.command()
def health() -> None:
    """Kiểm tra Postgres, pgvector, Ollama và các model."""
    checks = health_checks(get_settings())
    table = Table(show_header=False, box=None)
    for c in checks:
        mark = "[green]OK[/]" if c.ok else ("[red]FAIL[/]" if c.required else "[yellow]WARN[/]")
        table.add_row(c.name, mark, c.detail)
    console.print(table)
    if any(c.required and not c.ok for c in checks):
        raise typer.Exit(1)


@app.command()
def ingest(
    path: Annotated[Path, typer.Argument(help="File hoặc thư mục TXT/MD")],
    prune: Annotated[bool, typer.Option(help="Xóa document không còn trong thư mục")] = False,
    dry_run: Annotated[bool, typer.Option(help="Chỉ báo cáo, không embed/ghi DB")] = False,
) -> None:
    """Nạp tài liệu vào vector store (idempotent)."""
    engine = _engine()
    try:
        with console.status("Đang ingest..."):
            report = engine.pipeline.ingest(path, prune=prune, dry_run=dry_run)
    except (FileNotFoundError, EmbeddingError) as exc:
        console.print(f"[red]✗[/] {exc}")
        raise typer.Exit(1) from exc
    finally:
        engine.close()
    console.print(
        f"{'[dim](dry-run)[/] ' if dry_run else ''}"
        f"added={report.added} updated={report.updated} skipped={report.skipped} "
        f"deleted={report.deleted} chunks={report.chunks} elapsed={report.elapsed_ms / 1000:.1f}s"
    )


@app.command()
def stats() -> None:
    """Số documents, chunks và cấu hình embedding đã lưu."""
    engine = _engine()
    s = engine.store.stats()
    engine.close()
    console.print(" ".join(f"{k}={v}" for k, v in s.items()))


@app.command()
def reset(yes: Annotated[bool, typer.Option("--yes", "-y", help="Không hỏi xác nhận")] = False) -> None:
    """Xóa toàn bộ dữ liệu đã ingest."""
    if not yes:
        typer.confirm("Xóa toàn bộ documents/chunks?", abort=True)
    store = build_store(get_settings(), verify_meta=False)
    store.reset()
    store.close()
    console.print("[green]Đã reset.[/]")


@app.command()
def search(
    query: Annotated[str, typer.Argument(help="Câu truy vấn")],
    top_k: Annotated[int | None, typer.Option("--top-k", "-k")] = None,
    source: Annotated[str | None, typer.Option(help="Lọc theo tiền tố source")] = None,
    min_score: Annotated[float | None, typer.Option(help="Ghi đè RAG_MIN_SCORE")] = None,
    as_json: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Tìm kiếm ngữ nghĩa: in top-k chunk kèm score và nguồn."""
    engine = _engine()
    try:
        res = engine.retriever.retrieve(
            query, top_k or engine.settings.top_k, source_prefix=source, min_score=min_score
        )
    except (ValueError, EmbeddingError) as exc:
        console.print(f"[red]✗[/] {exc}")
        raise typer.Exit(1) from exc
    finally:
        engine.close()

    if as_json:
        print(SearchResponse.of(res).model_dump_json(indent=2))
        return
    if not res.results:
        console.print("[yellow]Không có chunk nào đạt ngưỡng min_score.[/]")
    table = Table(show_lines=False)
    table.add_column("#", justify="right")
    table.add_column("score", justify="right")
    table.add_column("source › heading", overflow="fold")
    table.add_column("preview", overflow="fold")
    for i, r in enumerate(res.results, 1):
        where = r.source + (" › " + " › ".join(r.heading_path) if r.heading_path else "")
        preview = " ".join(r.text.split())[:200]
        table.add_row(str(i), f"{r.score:.3f}", where, preview)
    if res.results:
        console.print(table)
    console.print(f"[dim]timings: {_fmt_ms(res.timings_ms)}[/]")


@app.command()
def ask(
    question: Annotated[str, typer.Argument(help="Câu hỏi")],
    top_k: Annotated[int | None, typer.Option("--top-k", "-k")] = None,
    source: Annotated[str | None, typer.Option(help="Lọc theo tiền tố source")] = None,
    min_score: Annotated[float | None, typer.Option(help="Ghi đè RAG_MIN_SCORE")] = None,
    show_context: Annotated[bool, typer.Option(help="In nội dung các đoạn context")] = False,
    as_json: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Hỏi đáp có trích dẫn nguồn [n]."""
    engine = _engine()
    try:
        with console.status(f"Đang hỏi {engine.settings.llm_model}..."):
            ans = engine.orchestrator.ask(
                question, top_k or engine.settings.top_k, source_prefix=source, min_score=min_score
            )
    except (ValueError, EmbeddingError, LLMError) as exc:
        console.print(f"[red]✗[/] {exc}")
        raise typer.Exit(1) from exc
    finally:
        engine.close()

    if as_json:
        print(AskResponse.of(ans).model_dump_json(indent=2))
        return

    console.print(ans.text if ans.answered else f"[yellow]{ans.text}[/]")
    if ans.sources:
        console.print("\n[bold]Sources:[/]")
        for i, s in enumerate(ans.sources, 1):
            mark = "[green]✓[/]" if i in ans.cited else " "
            console.print(f" {mark} [{i}] {format_source(s)} [dim]({s.score:.3f})[/]", highlight=False)
            if show_context:
                console.print(f"[dim]{s.text}[/]\n", highlight=False)
    if ans.invalid_citations:
        console.print(f"[red]⚠ Trích dẫn không hợp lệ: {ans.invalid_citations}[/]")
    elif ans.answered and not ans.cited:
        console.print("[yellow]⚠ Câu trả lời không có trích dẫn nào.[/]")
    console.print(f"[dim]timings: {_fmt_ms(ans.timings_ms)}[/]")


@app.command()
def serve(
    host: Annotated[
        str, typer.Option(help="Mặc định chỉ localhost: /ingest đọc được mọi đường dẫn trên máy")
    ] = "127.0.0.1",
    port: Annotated[int, typer.Option()] = 8000,
    reload: Annotated[bool, typer.Option(help="Tự reload khi sửa code (dev)")] = False,
) -> None:
    """Chạy REST API (FastAPI). Swagger UI tại http://HOST:PORT/docs"""
    import uvicorn

    console.print(f"Swagger UI: [link]http://{host}:{port}/docs[/link]")
    uvicorn.run("rag_engine.api.main:create_app", factory=True, host=host, port=port, reload=reload)


if __name__ == "__main__":
    app()
