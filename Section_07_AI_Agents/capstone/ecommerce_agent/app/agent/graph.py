"""StateGraph wiring.

    START -> router --(tool calls)---------> tools -> router
                    --(escalate / no spec)-> human_escalation -> END
                    --(final answer)-------> END
"""

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.agent.nodes import (
    ESCALATION_NODE,
    ROUTER_NODE,
    TOOLS_NODE,
    human_escalation_node,
    route_after_router,
    router_node,
    tool_executor_node,
)
from app.agent.state import AgentState


def build_graph(checkpointer: BaseCheckpointSaver | None = None) -> CompiledStateGraph:
    builder = StateGraph(AgentState)

    builder.add_node(ROUTER_NODE, router_node)
    builder.add_node(TOOLS_NODE, tool_executor_node)
    builder.add_node(ESCALATION_NODE, human_escalation_node)

    builder.add_edge(START, ROUTER_NODE)
    builder.add_conditional_edges(
        ROUTER_NODE,
        route_after_router,
        {TOOLS_NODE: TOOLS_NODE, ESCALATION_NODE: ESCALATION_NODE, END: END},
    )
    builder.add_edge(TOOLS_NODE, ROUTER_NODE)
    builder.add_edge(ESCALATION_NODE, END)

    return builder.compile(checkpointer=checkpointer)
