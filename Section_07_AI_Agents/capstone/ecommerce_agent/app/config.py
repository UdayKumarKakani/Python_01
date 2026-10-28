"""Application settings loaded from environment variables / `.env`."""

from functools import lru_cache

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Service -------------------------------------------------------------
    app_name: str = "ACME E-Commerce Support Agent"
    log_level: str = "INFO"
    # Log customer messages, LLM answers and tool output (truncated). Disable if logs must not hold PII.
    log_message_content: bool = True
    log_preview_chars: int = Field(200, ge=20)
    request_timeout_seconds: float = Field(90.0, gt=0)

    # --- LLM (pluggable) -----------------------------------------------------
    llm_provider: str = "together"  # together | openai | anthropic | self_hosted
    llm_model: str | None = None  # None -> provider default (see llm/factory.py)
    llm_temperature: float = Field(0.0, ge=0.0, le=2.0)
    llm_max_tokens: int = Field(1024, gt=0)
    llm_timeout_seconds: float = Field(30.0, gt=0)
    llm_max_retries: int = Field(2, ge=0)
    llm_base_url: str | None = None  # self_hosted: OpenAI-compatible endpoint (vLLM, Ollama, TGI)

    together_api_key: SecretStr | None = None
    openai_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None

    # --- Embeddings (must match the Pinecone index dimension) ----------------
    embedding_provider: str = "huggingface"  # huggingface (local) | together | openai
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dimension: int = 384
    embedding_device: str | None = None  # None -> auto (cuda / mps / cpu)
    embedding_local_files_only: bool = True  # never hit the network to load the local model
    embedding_query_instruction: str | None = "Represent this sentence for searching relevant passages: "

    # --- Pinecone ------------------------------------------------------------
    pinecone_api_key: SecretStr | None = None
    pinecone_index_name: str = "acme-knowledge-base"  # created on startup if missing
    pinecone_namespace: str | None = None
    pinecone_top_k: int = Field(4, ge=1, le=20)
    pinecone_min_score: float = Field(0.55, ge=0.0, le=1.0)  # tuned for BGE cosine scores
    pinecone_timeout_seconds: float = Field(10.0, gt=0)
    pinecone_cloud: str = "aws"  # used when the index has to be created
    pinecone_region: str = "us-east-1"

    # --- Agent graph ---------------------------------------------------------
    graph_recursion_limit: int = Field(12, ge=4)
    max_history_messages: int = Field(30, ge=4)

    # --- Conversation memory -------------------------------------------------
    checkpointer: str = "memory"  # memory | sqlite
    checkpoint_sqlite_path: str = "data/checkpoints.db"

    # --- Store DB (products + orders) ----------------------------------------
    # None -> share the checkpointer's SQLite file (separate tables).
    store_sqlite_path: str | None = None
    store_query_timeout_seconds: float = Field(5.0, gt=0)

    @field_validator("llm_provider", "embedding_provider", "checkpointer")
    @classmethod
    def _normalize_provider(cls, value: str) -> str:
        return value.strip().lower()

    @property
    def resolved_store_path(self) -> str:
        return self.store_sqlite_path or self.checkpoint_sqlite_path


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
