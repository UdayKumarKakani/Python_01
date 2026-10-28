"""LangGraph workflow state."""

from typing import Annotated, NotRequired, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    # Sticky per session: once True it stays True in the checkpoint.
    needs_human_escalation: NotRequired[bool]
    current_order_id: NotRequired[str | None]
    # For the human operator picking up the ticket.
    escalation_reason: NotRequired[str | None]
