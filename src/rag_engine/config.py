from functools import lru_cache
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Cấu hình engine, đọc từ biến môi trường `RAG_*` hoặc file `.env`."""

    model_config = SettingsConfigDict(env_prefix="RAG_", env_file=".env", extra="ignore")

    database_url: str = "postgresql://rag:rag@localhost:5432/rag"
    ollama_url: str = "http://localhost:11434"
    ollama_timeout_s: float = 300.0

    embed_model: str = "bge-m3"
    embed_dim: int = Field(1024, gt=0)
    embed_batch_size: int = Field(16, gt=0)
    embed_num_gpu: int | None = Field(
        None, ge=0, description="Số layer embed đưa lên GPU; 0 = chạy CPU, None = Ollama tự quyết"
    )

    llm_model: str = "qwen2.5:3b-instruct"
    llm_temperature: float = 0.1
    llm_num_ctx: int = Field(4096, gt=0, description="Context window gửi cho Ollama")

    chunk_size: int = Field(512, gt=0, description="Kích thước chunk tối đa (token ước lượng)")
    chunk_overlap: int = Field(64, ge=0)
    chars_per_token: float = Field(4.0, gt=0, description="Hệ số ước lượng ký tự/token")
    min_section_tokens: int = Field(50, ge=0, description="Section nhỏ hơn sẽ gộp vào section sau")

    top_k: int = Field(5, gt=0)
    min_score: float = Field(0.50, ge=-1.0, le=1.0)
    hnsw_ef_search: int = Field(40, gt=0)
    search_mode: Literal["auto", "dense", "hybrid"] = Field(
        "auto",
        description="dense = chỉ vector; hybrid = cosine + hybrid_alpha × độ phủ từ khóa; "
        "auto = dense khi bật rerank (reranker tự chọn lại tốt hơn), hybrid khi không (ADR-0011)",
    )
    hybrid_alpha: float = Field(
        0.15, ge=0.0, description="Trọng số độ phủ từ khóa (IDF, 0–1) cộng vào cosine khi xếp hạng"
    )
    rerank_enabled: bool = Field(False, description="Bật bước rerank cross-encoder (cần service reranker)")
    rerank_url: str = "http://localhost:8081"
    rerank_candidates: int = Field(
        20, gt=0, description="Số chunk đưa vào reranker; trả về top_k theo điểm rerank"
    )
    rerank_timeout_s: float = 60.0
    hybrid_candidates: int = Field(
        20, gt=0, description="Số ứng viên lấy từ mỗi nhánh (dense, full-text) trước khi gộp"
    )
    context_max_chars: int = Field(8000, gt=0, description="Budget context; giữ dưới ~num_ctx*2.5 ký tự")

    log_level: str = "INFO"

    @property
    def effective_search_mode(self) -> Literal["dense", "hybrid"]:
        if self.search_mode != "auto":
            return self.search_mode
        return "dense" if self.rerank_enabled else "hybrid"

    @model_validator(mode="after")
    def _check_overlap(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap phải nhỏ hơn chunk_size")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
