"""Graph nodes and routing.

Nodes never construct an LLM client themselves; they get one from
`app.llm.get_chat_model()`, so the provider is purely a config concern.
"""

import asyncio
import logging
import time
from functools import lru_cache
from typing import Literal

from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    trim_messages,
)
from langchain_core.runnables import Runnable, RunnableConfig
from langgraph.graph import END
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel, Field

from app.agent.prompts import HUMAN_HANDOFF_MESSAGE, build_system_prompt, is_unverified_spec_response
from app.agent.state import AgentState
from app.config import get_settings
from app.exceptions import AgentServiceError, LLMProviderError, LLMTimeoutError
from app.llm import get_chat_model
from app.logging_config import preview
from app.tools import CUSTOMER_TOOLS
from app.tools.order_lookup import is_successful_lookup, normalize_order_id

logger = logging.getLogger(__name__)

ROUTER_NODE = "router"
TOOLS_NODE = "tools"
ESCALATION_NODE = "human_escalation"


class escalate_to_human(BaseModel):  # noqa: N801 - class name is the tool name the LLM sees
    """Hand the conversation to a human support agent. Call this immediately when the customer
    is extremely angry, disputes a charge or transaction, reports a broken/damaged/defective
    item, asks for a human, or a store system is unavailable."""

    reason: Literal["extreme_anger", "transaction_dispute", "defective_item", "requested_human", "system_issue", "other"]
    summary: str = Field(..., description="One sentence summarizing the issue for the human agent.")


ESCALATION_TOOL_NAME = escalate_to_human.__name__


def message_text(message: BaseMessage) -> str:
    """Plain text of a message, whether content is a string or a list of content blocks."""
    content = message.content
    if isinstance(content, str):
        return content
    return "".join(
        block if isinstance(block, str) else block.get("text", "")
        for block in content
        if isinstance(block, str) or block.get("type") == "text"
    )


def current_turn(messages: list[AnyMessage]) -> list[AnyMessage]:
    """Messages produced since the customer's latest message (inclusive)."""
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], HumanMessage):
            return messages[i:]
    return messages


def describe_tool_calls(tool_calls: list[dict]) -> str:
    return ", ".join(
        f"{c['name']}({', '.join(f'{k}={v!r}' for k, v in c['args'].items())})" for c in tool_calls
    )


@lru_cache(maxsize=1)
def _get_router_model() -> Runnable:
    # `escalate_to_human` is bound as a routing signal only; it is never executed by ToolNode.
    return get_chat_model().bind_tools([*CUSTOMER_TOOLS, escalate_to_human])


def _close_dangling_tool_calls(messages: list[AnyMessage]) -> list[AnyMessage]:
    """Every AI tool call must be followed by a ToolMessage or providers reject the request.
    A turn interrupted by a timeout/recursion limit can leave one open in the checkpoint."""
    answered = {m.tool_call_id for m in messages if isinstance(m, ToolMessage)}
    repaired: list[AnyMessage] = []
    for message in messages:
        repaired.append(message)
        if isinstance(message, AIMessage):
            repaired.extend(
                ToolMessage(content="This action was not completed.", tool_call_id=call["id"], name=call["name"])
                for call in message.tool_calls
                if call["id"] not in answered
            )
    return repaired


def _prepare_history(messages: list[AnyMessage], max_messages: int) -> list[AnyMessage]:
    return trim_messages(
        _close_dangling_tool_calls(messages),
        max_tokens=max_messages,
        token_counter=len,  # count messages, not tokens
        strategy="last",
        start_on="human",
        include_system=False,
    )


# --- Nodes -------------------------------------------------------------------


async def router_node(state: AgentState) -> dict:
    """LLM decides: answer directly, call a tool, or escalate."""
    settings = get_settings()
    history = _prepare_history(state["messages"], settings.max_history_messages)
    prompt = [SystemMessage(content=build_system_prompt(state.get("current_order_id"))), *history]

    call_number = sum(isinstance(m, AIMessage) for m in current_turn(state["messages"])) + 1
    logger.info(
        "[router] LLM call #%d: %d messages (system + %d of %d in history), active order=%s",
        call_number,
        len(prompt),
        len(history),
        len(state["messages"]),
        state.get("current_order_id") or "none",
    )
    started = time.perf_counter()

    try:
        response = await asyncio.wait_for(
            _get_router_model().ainvoke(prompt),
            timeout=settings.llm_timeout_seconds,
        )
    except TimeoutError as exc:
        logger.warning("LLM call exceeded %.1fs", settings.llm_timeout_seconds)
        raise LLMTimeoutError("LLM call timed out") from exc
    except AgentServiceError:
        raise
    except Exception as exc:
        logger.exception("LLM provider call failed")
        raise LLMProviderError(f"LLM provider error: {exc}") from exc

    elapsed_ms = (time.perf_counter() - started) * 1000
    usage = response.usage_metadata or {}
    tokens = f"tokens in={usage.get('input_tokens', '?')} out={usage.get('output_tokens', '?')}"
    if response.tool_calls:
        decision = f"tool calls: {preview(describe_tool_calls(response.tool_calls))}"
    else:
        decision = f"answer: {preview(message_text(response))}"
    logger.info("[router] LLM call #%d replied in %.0f ms (%s) -> %s", call_number, elapsed_ms, tokens, decision)

    return {"messages": [response]}


def _tool_error_message(error: Exception) -> str:
    logger.error("Tool execution failed: %s", error)
    return (
        "TOOL_ERROR: The tool could not complete this request. Do not guess the answer. "
        "If the customer's question depends on it, offer to connect them with a team member."
    )


_tool_node = ToolNode(CUSTOMER_TOOLS, handle_tool_errors=_tool_error_message)


async def tool_executor_node(state: AgentState, config: RunnableConfig) -> dict:
    """Run requested tools and remember the order ID of any successful lookup."""
    tool_calls = state["messages"][-1].tool_calls
    logger.info("[tools] running %d tool call(s)%s", len(tool_calls), " in parallel" if len(tool_calls) > 1 else "")
    for call in tool_calls:
        logger.info("[tools] -> %s", preview(describe_tool_calls([call])))

    started = time.perf_counter()
    result = await _tool_node.ainvoke(state, config)
    elapsed_ms = (time.perf_counter() - started) * 1000
    tool_messages: list[ToolMessage] = result["messages"]
    update: dict = {"messages": tool_messages}

    calls_by_id = {call["id"]: call for call in tool_calls}
    for msg in tool_messages:
        call = calls_by_id.get(msg.tool_call_id)
        text = message_text(msg)
        logger.info("[tools] <- %s %s: %s", msg.name, "FAILED" if msg.status == "error" else "ok", preview(text))
        if call and call["name"] == "track_order" and msg.status != "error" and is_successful_lookup(text):
            update["current_order_id"] = normalize_order_id(call["args"]["order_id"])
            if update["current_order_id"] != state.get("current_order_id"):
                logger.info("[tools] active order set to %s", update["current_order_id"])

    logger.info("[tools] finished in %.0f ms", elapsed_ms)
    return update


async def human_escalation_node(state: AgentState) -> dict:
    """Flag the session for a human operator and close out any pending tool calls."""
    last = state["messages"][-1]
    new_messages: list[AnyMessage] = []
    reason, summary = "unverified_specification", "Customer asked for a spec not present in the catalog."

    if isinstance(last, AIMessage) and last.tool_calls:
        for call in last.tool_calls:
            if call["name"] == ESCALATION_TOOL_NAME:
                reason = call["args"].get("reason", "other")
                summary = call["args"].get("summary", "")
                content = "Escalated to a human agent."
            else:
                content = "Skipped: conversation was escalated to a human agent."
            new_messages.append(ToolMessage(content=content, tool_call_id=call["id"], name=call["name"]))
        new_messages.append(AIMessage(content=HUMAN_HANDOFF_MESSAGE))
    # Otherwise the last AI message already is the customer-facing unverified-spec reply.

    # Integration point: create a helpdesk ticket / push to the live-agent queue here.
    logger.warning(
        "[escalation] handing off to a human: reason=%s order=%s summary=%s",
        reason,
        state.get("current_order_id") or "none",
        preview(summary),
    )
    return {"messages": new_messages, "needs_human_escalation": True, "escalation_reason": reason}


# --- Routing -----------------------------------------------------------------


def route_after_router(state: AgentState) -> Literal["tools", "human_escalation", "__end__"]:
    last = state["messages"][-1]

    if isinstance(last, AIMessage) and any(c["name"] == ESCALATION_TOOL_NAME for c in last.tool_calls):
        route, why = ESCALATION_NODE, "model called escalate_to_human"
    elif tools_condition(state) == TOOLS_NODE:
        route, why = TOOLS_NODE, f"{len(last.tool_calls)} tool call(s) requested"
    elif isinstance(last, AIMessage) and is_unverified_spec_response(message_text(last)):
        route, why = ESCALATION_NODE, "unverified-spec reply detected"
    else:
        route, why = END, "final answer"

    logger.info("[route] router -> %s (%s)", "END" if route == END else route, why)
    return route
