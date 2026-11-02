# RAG_Engine
A reusable, modular Retrieval-Augmented Generation (RAG) engine providing document ingestion, text chunking, embedding generation, vector storage, semantic retrieval, reranking, and LLM-powered response generation. Designed to be extensible, scalable, and reusable across multiple applications and domains.

## Documentation
Design docs (architecture, milestones, ADRs, local setup) live in [`Docs/`](Docs/README.md).

## Quick start
```bash
docker compose up -d
docker compose exec ollama ollama pull bge-m3
docker compose exec ollama ollama pull qwen2.5:3b-instruct
uv sync && uv run rag health
uv run rag ingest samples/
uv run rag ask "Your question here"
uv run rag serve        # REST API + Swagger UI at http://127.0.0.1:8000/docs
```
