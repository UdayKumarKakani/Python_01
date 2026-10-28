"""Pinecone client and index management, shared by the app and the ingestion script."""

import logging
import time
from functools import lru_cache
from typing import Any

import httpx
from pinecone import Pinecone, ServerlessSpec

from app.config import Settings, get_settings
from app.exceptions import VectorStoreError

logger = logging.getLogger(__name__)

_INDEX_READY_TIMEOUT_SECONDS = 180
# Newer API version that exposes an index's document schema (older client versions can't see it).
_SCHEMA_API_VERSION = "2026-07"


def _integrated_embedding_fields(name: str) -> list[str]:
    """Fields Pinecone embeds itself. Such indexes only accept text via the documents API,
    not the vectors this app produces. Best effort: returns [] if the check can't run."""
    settings = get_settings()
    try:
        response = httpx.get(
            f"https://api.pinecone.io/indexes/{name}",
            headers={
                "Api-Key": settings.pinecone_api_key.get_secret_value(),
                "X-Pinecone-API-Version": _SCHEMA_API_VERSION,
            },
            timeout=settings.pinecone_timeout_seconds,
        )
        response.raise_for_status()
        fields = (response.json().get("schema") or {}).get("fields") or {}
    except Exception as exc:
        logger.debug("Could not inspect schema of index '%s': %s", name, exc)
        return []
    return [field for field, spec in fields.items() if isinstance(spec, dict) and spec.get("embed")]


@lru_cache(maxsize=1)
def get_pinecone_client() -> Pinecone:
    settings = get_settings()
    if settings.pinecone_api_key is None or not settings.pinecone_api_key.get_secret_value().strip():
        raise VectorStoreError("PINECONE_API_KEY is not set")
    return Pinecone(api_key=settings.pinecone_api_key.get_secret_value())


def _is_ready(description: Any) -> bool:
    status = description.status
    return bool(status.get("ready") if isinstance(status, dict) else getattr(status, "ready", False))


def ensure_index(settings: Settings | None = None) -> None:
    """Create the index if it doesn't exist; otherwise check its dimension matches the embedder.

    Blocking (Pinecone's client is sync). From async code use `asyncio.to_thread(ensure_index)`.
    """
    settings = settings or get_settings()
    client = get_pinecone_client()
    name = settings.pinecone_index_name

    if name not in client.list_indexes().names():
        logger.info("Creating Pinecone index '%s' (dim=%d, cosine)", name, settings.embedding_dimension)
        client.create_index(
            name=name,
            dimension=settings.embedding_dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud=settings.pinecone_cloud, region=settings.pinecone_region),
        )
        deadline = time.monotonic() + _INDEX_READY_TIMEOUT_SECONDS
        while not _is_ready(client.describe_index(name)):
            if time.monotonic() > deadline:
                raise VectorStoreError(f"Index '{name}' was not ready after {_INDEX_READY_TIMEOUT_SECONDS}s")
            time.sleep(2)
        logger.info("Pinecone index '%s' is ready", name)
        return

    embedded_fields = _integrated_embedding_fields(name)
    if embedded_fields:
        raise VectorStoreError(
            f"Index '{name}' is an integrated-embedding index (Pinecone embeds field(s) "
            f"{embedded_fields} itself), so it can't store this app's vectors. "
            "Set PINECONE_INDEX_NAME to a new name and it will be created as a standard index."
        )

    dimension = client.describe_index(name).dimension
    if dimension != settings.embedding_dimension:
        raise VectorStoreError(
            f"Index '{name}' has dimension {dimension} but EMBEDDING_DIMENSION is "
            f"{settings.embedding_dimension}. Use a different PINECONE_INDEX_NAME or embedding model."
        )
    logger.info("Using existing Pinecone index '%s' (dim=%d)", name, dimension)


@lru_cache(maxsize=1)
def get_index():
    return get_pinecone_client().Index(get_settings().pinecone_index_name)
