"""Pluggable LLM gateway.

The rest of the app only ever sees `BaseChatModel` / `Embeddings`. Switching
providers is a config change (`LLM_PROVIDER=openai`), never a graph change.

Adding a provider:
    1. Write a `_build_<name>(settings) -> BaseChatModel` function.
    2. Register it in `_CHAT_BUILDERS` (or call `register_chat_provider` at startup).

Provider SDKs are imported lazily so only the one you use needs to be installed.
"""

import logging
from collections.abc import Callable

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import SecretStr

from app.config import Settings, get_settings
from app.exceptions import LLMConfigurationError

logger = logging.getLogger(__name__)

ChatModelBuilder = Callable[[Settings], BaseChatModel]
EmbeddingsBuilder = Callable[[Settings], Embeddings]

DEFAULT_CHAT_MODELS: dict[str, str] = {
    "together": "nvidia/NVIDIA-Nemotron-3.5-Super-VL",
    "openai": "gpt-4o-mini",
    "anthropic": "claude-sonnet-5-5",
}


def _require_key(secret: SecretStr | None, env_name: str) -> str:
    if secret is None or not secret.get_secret_value().strip():
        raise LLMConfigurationError(f"{env_name} is not set")
    return secret.get_secret_value()


def _model_name(settings: Settings) -> str:
    name = settings.llm_model or DEFAULT_CHAT_MODELS.get(settings.llm_provider)
    if not name:
        raise LLMConfigurationError(f"LLM_MODEL must be set for provider '{settings.llm_provider}'")
    return name


# --- Chat model builders -----------------------------------------------------


def _build_together(settings: Settings) -> BaseChatModel:
    from langchain_together import ChatTogether

    return ChatTogether(
        model=_model_name(settings),
        api_key=_require_key(settings.together_api_key, "TOGETHER_API_KEY"),
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        timeout=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
    )


def _build_openai(settings: Settings) -> BaseChatModel:
    # pip install langchain-openai
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=_model_name(settings),
        api_key=_require_key(settings.openai_api_key, "OPENAI_API_KEY"),
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        timeout=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
    )


def _build_anthropic(settings: Settings) -> BaseChatModel:
    # pip install langchain-anthropic
    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(
        model=_model_name(settings),
        api_key=_require_key(settings.anthropic_api_key, "ANTHROPIC_API_KEY"),
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        timeout=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
    )


def _build_self_hosted(settings: Settings) -> BaseChatModel:
    """Any OpenAI-compatible server (vLLM, Ollama, TGI, LM Studio)."""
    from langchain_openai import ChatOpenAI

    if not settings.llm_base_url:
        raise LLMConfigurationError("LLM_BASE_URL must be set for provider 'self_hosted'")
    api_key = settings.openai_api_key.get_secret_value() if settings.openai_api_key else "not-needed"
    return ChatOpenAI(
        model=_model_name(settings),
        base_url=settings.llm_base_url,
        api_key=api_key,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        timeout=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
    )


_CHAT_BUILDERS: dict[str, ChatModelBuilder] = {
    "together": _build_together,
    "openai": _build_openai,
    "anthropic": _build_anthropic,
    "self_hosted": _build_self_hosted,
}


def register_chat_provider(name: str, builder: ChatModelBuilder) -> None:
    """Extension point for providers not shipped with the service."""
    _CHAT_BUILDERS[name.strip().lower()] = builder


def get_chat_model(settings: Settings | None = None) -> BaseChatModel:
    """Return a configured chat model for `LLM_PROVIDER`.

    Builds a new instance on each call; callers that invoke it repeatedly
    should cache the result (see `app.agent.nodes`).
    """
    settings = settings or get_settings()
    builder = _CHAT_BUILDERS.get(settings.llm_provider)
    if builder is None:
        raise LLMConfigurationError(
            f"Unknown LLM_PROVIDER '{settings.llm_provider}'. Available: {sorted(_CHAT_BUILDERS)}"
        )

    try:
        model = builder(settings)
    except ImportError as exc:
        raise LLMConfigurationError(
            f"SDK for provider '{settings.llm_provider}' is not installed: {exc}"
        ) from exc

    # The agent depends on tool calling; fail fast at startup rather than mid-conversation.
    if type(model).bind_tools is BaseChatModel.bind_tools:
        raise LLMConfigurationError(f"{type(model).__name__} does not support .bind_tools()")

    logger.info("Initialized chat model provider=%s model=%s", settings.llm_provider, _model_name(settings))
    return model


# --- Embeddings --------------------------------------------------------------


def _build_together_embeddings(settings: Settings) -> Embeddings:
    from langchain_together import TogetherEmbeddings

    return TogetherEmbeddings(
        model=settings.embedding_model,
        api_key=_require_key(settings.together_api_key, "TOGETHER_API_KEY"),
        timeout=settings.pinecone_timeout_seconds,
        max_retries=settings.llm_max_retries,
    )


def _build_openai_embeddings(settings: Settings) -> Embeddings:
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=_require_key(settings.openai_api_key, "OPENAI_API_KEY"),
        dimensions=settings.embedding_dimension,
        timeout=settings.pinecone_timeout_seconds,
        max_retries=settings.llm_max_retries,
    )


def _build_huggingface_embeddings(settings: Settings) -> Embeddings:
    """Local sentence-transformers model (e.g. BGE) — no API key, runs on CPU / MPS / CUDA."""
    # pip install langchain-huggingface sentence-transformers
    from langchain_huggingface import HuggingFaceEmbeddings

    model_kwargs: dict = {"local_files_only": settings.embedding_local_files_only}
    if settings.embedding_device:
        model_kwargs["device"] = settings.embedding_device

    # BGE retrieval works best with an instruction on queries (not on documents).
    query_kwargs: dict = {"normalize_embeddings": True}
    if settings.embedding_query_instruction:
        query_kwargs["prompt"] = settings.embedding_query_instruction

    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs=model_kwargs,
            encode_kwargs={"normalize_embeddings": True},
            query_encode_kwargs=query_kwargs,
        )
    except OSError as exc:  # not in the local Hugging Face cache
        raise LLMConfigurationError(
            f"Embedding model '{settings.embedding_model}' was not found locally. Download it once with "
            "EMBEDDING_LOCAL_FILES_ONLY=false, or pick a model that is already cached."
        ) from exc

    dimension = len(embeddings.embed_documents(["dimension check"])[0])
    if dimension != settings.embedding_dimension:
        raise LLMConfigurationError(
            f"'{settings.embedding_model}' produces {dimension}-dim vectors but EMBEDDING_DIMENSION is "
            f"{settings.embedding_dimension}"
        )
    logger.info("Loaded local embedding model %s (dim=%d)", settings.embedding_model, dimension)
    return embeddings


_EMBEDDING_BUILDERS: dict[str, EmbeddingsBuilder] = {
    "huggingface": _build_huggingface_embeddings,
    "together": _build_together_embeddings,
    "openai": _build_openai_embeddings,
}


def get_embedding_model(settings: Settings | None = None) -> Embeddings:
    settings = settings or get_settings()
    builder = _EMBEDDING_BUILDERS.get(settings.embedding_provider)
    if builder is None:
        raise LLMConfigurationError(
            f"Unknown EMBEDDING_PROVIDER '{settings.embedding_provider}'. Available: {sorted(_EMBEDDING_BUILDERS)}"
        )
    try:
        return builder(settings)
    except ImportError as exc:
        raise LLMConfigurationError(f"SDK for embeddings provider is not installed: {exc}") from exc
