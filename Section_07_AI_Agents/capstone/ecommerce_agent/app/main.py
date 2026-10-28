"""FastAPI entrypoint."""

import asyncio
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.errors import GraphRecursionError
from pydantic import BaseModel, ConfigDict, Field

from app.agent import build_graph
from app.agent.checkpointer import open_checkpointer
from app.agent.nodes import ESCALATION_NODE, current_turn, message_text
from app.agent.prompts import HUMAN_HANDOFF_MESSAGE
from app.config import get_settings
from app.db import init_db
from app.exceptions import AgentServiceError, LLMTimeoutError
from app.logging_config import preview, request_id_var, session_id_var, setup_logging
from app.tools.vector_search import get_embedder
from app.vectorstore import ensure_index

# The local embedding model's tokenizer warns (and disables itself) if the process forks after
# it has run; we don't rely on its internal parallelism, so turn it off up front.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Configure logging at import so startup (model loading, index checks) is visible too.
setup_logging(get_settings().log_level)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    product_count = await init_db(settings)
    if product_count == 0:
        logger.warning("Store DB has no products. Run `python -m scripts.seed_db`.")

    # Create the Pinecone index if missing. A failure here must not block startup:
    # the knowledge base tool reports KB_UNAVAILABLE and the agent escalates.
    if settings.pinecone_api_key is not None:
        try:
            await asyncio.to_thread(ensure_index, settings)
            # Load the embedding model now so the first customer question doesn't pay for it.
            await asyncio.to_thread(get_embedder)
        except Exception:
            logger.exception("Knowledge base setup failed; knowledge base search will be unavailable")
    else:
        logger.warning("PINECONE_API_KEY is not set; knowledge base search will be unavailable")

    async with open_checkpointer(settings) as checkpointer:
        app.state.graph = build_graph(checkpointer=checkpointer)
        logger.info(
            "Agent graph compiled (provider=%s, checkpointer=%s)", settings.llm_provider, settings.checkpointer
        )
        yield


app = FastAPI(title=get_settings().app_name, version="1.0.0", lifespan=lifespan)


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    """Tag every log line of this request with an ID; honor X-Request-ID from a proxy if present."""
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:8]
    request_id_var.set(request_id)
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


# --- Schemas -----------------------------------------------------------------


class ChatRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    session_id: str = Field(..., min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.:\-]+$")
    message: str = Field(..., min_length=1, max_length=4000)


class ChatResponse(BaseModel):
    session_id: str
    message: str
    needs_human_escalation: bool
    current_order_id: str | None = None


class ErrorResponse(BaseModel):
    error: str
    detail: str


# --- Error handlers ----------------------------------------------------------


@app.exception_handler(AgentServiceError)
async def agent_error_handler(request: Request, exc: AgentServiceError) -> JSONResponse:
    logger.error("%s on %s: %s", type(exc).__name__, request.url.path, exc)
    headers = {"Retry-After": "5"} if exc.status_code in (503, 504) else None
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(error=exc.error_code, detail=exc.public_message).model_dump(),
        headers=headers,
    )


@app.exception_handler(Exception)
async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled error on %s", request.url.path)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(error="internal_error", detail=AgentServiceError.public_message).model_dump(),
    )


# --- Routes ------------------------------------------------------------------


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post(
    "/api/v1/chat",
    response_model=ChatResponse,
    responses={500: {"model": ErrorResponse}, 503: {"model": ErrorResponse}, 504: {"model": ErrorResponse}},
)
async def chat(payload: ChatRequest, request: Request) -> ChatResponse:
    settings = get_settings()
    session_id_var.set(payload.session_id)
    logger.info("[chat] user message: %s", preview(payload.message))
    started = time.perf_counter()

    graph = request.app.state.graph
    config = {
        "configurable": {"thread_id": payload.session_id},
        "recursion_limit": settings.graph_recursion_limit,
    }

    try:
        state = await asyncio.wait_for(
            graph.ainvoke({"messages": [HumanMessage(content=payload.message)]}, config=config),
            timeout=settings.request_timeout_seconds,
        )
    except TimeoutError as exc:
        logger.error("[chat] turn exceeded REQUEST_TIMEOUT_SECONDS=%s", settings.request_timeout_seconds)
        raise LLMTimeoutError(f"Request exceeded {settings.request_timeout_seconds}s") from exc
    except GraphRecursionError:
        # The agent looped on tools without converging: hand off instead of failing.
        logger.error("[chat] recursion limit (%d steps) hit; escalating", settings.graph_recursion_limit)
        await graph.aupdate_state(
            config,
            {
                "messages": [AIMessage(content=HUMAN_HANDOFF_MESSAGE)],
                "needs_human_escalation": True,
                "escalation_reason": "agent_loop_limit",
            },
            as_node=ESCALATION_NODE,
        )
        snapshot = await graph.aget_state(config)
        return ChatResponse(
            session_id=payload.session_id,
            message=HUMAN_HANDOFF_MESSAGE,
            needs_human_escalation=True,
            current_order_id=snapshot.values.get("current_order_id"),
        )

    reply = next(
        (message_text(m) for m in reversed(state["messages"]) if isinstance(m, AIMessage) and message_text(m)),
        HUMAN_HANDOFF_MESSAGE,
    )
    turn = current_turn(state["messages"])
    tools_used = [m.name for m in turn if isinstance(m, ToolMessage)]
    logger.info(
        "[chat] done in %.0f ms: llm_calls=%d tools=%s escalated=%s order=%s",
        (time.perf_counter() - started) * 1000,
        sum(isinstance(m, AIMessage) and m.content != HUMAN_HANDOFF_MESSAGE for m in turn),
        tools_used or "none",
        state.get("needs_human_escalation", False),
        state.get("current_order_id") or "none",
    )
    logger.info("[chat] final answer: %s", preview(reply))
    return ChatResponse(
        session_id=payload.session_id,
        message=reply,
        needs_human_escalation=state.get("needs_human_escalation", False),
        current_order_id=state.get("current_order_id"),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)
