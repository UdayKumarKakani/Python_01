"""Knowledge base RAG tool: Acme policies and help content stored in Pinecone."""

import asyncio
import logging
import time
from functools import lru_cache
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from app.config import get_settings
from app.exceptions import AgentServiceError, VectorStoreError
from app.llm.factory import get_embedding_model
from app.vectorstore import get_index

logger = logging.getLogger(__name__)


class KnowledgeBaseSearchInput(BaseModel):
    query: str = Field(
        ...,
        min_length=2,
        max_length=500,
        description="The policy or help question, e.g. 'return window for opened electronics'.",
    )


class KnowledgeChunk(BaseModel):
    id: str
    score: float
    text: str
    source: str | None = None
    page: int | None = None


@lru_cache(maxsize=1)
def get_embedder() -> Embeddings:
    return get_embedding_model()


async def search_knowledge_base(query: str) -> list[KnowledgeChunk]:
    """Embed `query` and run a similarity search. Raises VectorStoreError on any failure."""
    settings = get_settings()
    timeout = settings.pinecone_timeout_seconds

    started = time.perf_counter()
    try:
        # Loading a local model is slow and blocking: keep it off the event loop (cached after first call).
        embedder = await asyncio.to_thread(get_embedder)
        vector = await asyncio.wait_for(embedder.aembed_query(query), timeout=timeout)
    except TimeoutError as exc:
        raise VectorStoreError("Embedding request timed out") from exc
    except AgentServiceError:
        raise
    except Exception as exc:
        raise VectorStoreError(f"Embedding request failed: {exc}") from exc

    embed_ms = (time.perf_counter() - started) * 1000
    query_kwargs: dict[str, Any] = {"vector": vector, "top_k": settings.pinecone_top_k, "include_metadata": True}
    if settings.pinecone_namespace:
        query_kwargs["namespace"] = settings.pinecone_namespace

    started = time.perf_counter()
    try:
        index = get_index()
        # The Pinecone client is synchronous; run it off the event loop.
        response = await asyncio.wait_for(asyncio.to_thread(index.query, **query_kwargs), timeout=timeout)
    except TimeoutError as exc:
        raise VectorStoreError("Pinecone query timed out") from exc
    except AgentServiceError:
        raise
    except Exception as exc:
        raise VectorStoreError(f"Pinecone query failed: {exc}") from exc

    query_ms = (time.perf_counter() - started) * 1000
    chunks = []
    for match in response.matches:
        meta = dict(match.metadata or {})
        if (match.score or 0.0) < settings.pinecone_min_score or not meta.get("text"):
            continue
        page = meta.get("page")
        chunks.append(
            KnowledgeChunk(
                id=match.id,
                score=float(match.score),
                text=str(meta["text"]),
                source=meta.get("source"),
                page=int(page) if page is not None else None,
            )
        )

    hits = ", ".join(
        f"p{int((m.metadata or {}).get('page', 0))}={m.score or 0:.2f}" for m in response.matches
    )
    logger.info(
        "[kb] embed %.0f ms, Pinecone %.0f ms: %d matches [%s], %d kept (min_score=%.2f)",
        embed_ms,
        query_ms,
        len(response.matches),
        hits,
        len(chunks),
        settings.pinecone_min_score,
    )
    return chunks


def _format_chunk(rank: int, chunk: KnowledgeChunk) -> str:
    where = ", ".join(p for p in (chunk.source, f"page {chunk.page}" if chunk.page else None) if p)
    return f"[Passage {rank}] ({where}; relevance {chunk.score:.2f})\n{chunk.text}"


@tool("knowledge_base_search", args_schema=KnowledgeBaseSearchInput)
async def knowledge_base_search(query: str) -> str:
    """Search Acme's official help and policy documents: shipping options and costs, delivery
    times, returns and refunds, cancellations, warranties, Acme Plus membership, payment methods,
    pre-orders, price adjustments and how to contact support."""
    try:
        chunks = await search_knowledge_base(query)
    except AgentServiceError as exc:  # VectorStoreError, or a misconfigured embedding model
        logger.error("Knowledge base search failed query=%r: %s", query, exc)
        return (
            "KB_UNAVAILABLE: The policy knowledge base could not be reached. "
            "Do not answer policy questions from memory. Apologize and offer to connect the customer with a team member."
        )

    if not chunks:
        return (
            "NO_RELEVANT_POLICY: No policy passages matched this question. "
            "Tell the customer you couldn't find that in Acme's policies and offer to connect them with a team member."
        )
    return "\n\n".join(_format_chunk(i, c) for i, c in enumerate(chunks, start=1))
