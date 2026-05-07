"""
agent.py — LangGraph Agent using GROQ API
"""

import json
import os
import requests
import operator
from typing import Any, TypedDict, Annotated

from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
    ToolMessage
)
from langchain_core.tools import tool
from duckduckgo_search import DDGS


class AgentState(TypedDict):
    messages: Annotated[list, operator.add]


@tool
def web_search(query: str) -> str:
    """Search the web using DuckDuckGo."""
    try:
        results = DDGS().text(query, max_results=5)
        formatted = []

        for r in results:
            formatted.append(
                f"Title: {r.get('title')}\n"
                f"Snippet: {r.get('body')}\n"
                f"Link: {r.get('href')}\n"
            )

        return "\n\n".join(formatted)

    except Exception as e:
        return f"Search Error: {str(e)}"


@tool
def get_weather(city: str) -> str:
    """Get weather for any city."""
    try:
        geo = requests.get(
            f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1"
        ).json()

        result = geo["results"][0]

        lat = result["latitude"]
        lon = result["longitude"]

        weather = requests.get(
            f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
        ).json()

        current = weather["current_weather"]

        return (
            f"Weather in {city}\n"
            f"Temperature: {current['temperature']}°C\n"
            f"Wind Speed: {current['windspeed']} km/h"
        )

    except Exception as e:
        return f"Weather Error: {str(e)}"


@tool
def wikipedia_summary(topic: str) -> str:
    """Get Wikipedia summary."""
    try:
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{topic}"

        data = requests.get(url).json()

        return data.get("extract", "No summary found.")

    except Exception as e:
        return f"Wikipedia Error: {str(e)}"


TOOLS = [web_search, get_weather, wikipedia_summary]

TOOL_MAP = {t.name: t for t in TOOLS}


def get_llm():
    return ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="qwen/qwen3-32b",
        temperature=0
    ).bind_tools(TOOLS)


def agent_node(state: AgentState):
    llm = get_llm()

    response = llm.invoke(state["messages"])

    return {"messages": [response]}


def tool_node(state: AgentState):
    last_message = state["messages"][-1]

    outputs = []

    for tool_call in last_message.tool_calls:

        tool_fn = TOOL_MAP[tool_call["name"]]

        result = tool_fn.invoke(tool_call["args"])

        outputs.append(
            ToolMessage(
                content=str(result),
                tool_call_id=tool_call["id"]
            )
        )

    return {"messages": outputs}


def should_continue(state: AgentState):

    last = state["messages"][-1]

    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"

    return END


def build_graph():

    graph = StateGraph(AgentState)

    graph.add_node("agent", agent_node)

    graph.add_node("tools", tool_node)

    graph.set_entry_point("agent")

    graph.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            END: END
        }
    )

    graph.add_edge("tools", "agent")

    return graph.compile()


AGENT = build_graph()


SYSTEM_PROMPT = """
You are an intelligent AI Research Assistant.

Available Tools:
1. web_search
2. get_weather
3. wikipedia_summary

Use tools whenever needed.
"""


def run_agent(user_message, chat_history):

    messages = [SystemMessage(content=SYSTEM_PROMPT)]

    for msg in chat_history:

        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))

        else:
            messages.append(AIMessage(content=msg["content"]))

    messages.append(HumanMessage(content=user_message))

    result = AGENT.invoke({"messages": messages})

    final_response = result["messages"][-1].content

    updated_history = chat_history + [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": final_response}
    ]

    tool_steps = []

    def _attr(obj: Any, name: str):
        return getattr(obj, name, obj[name])

    for msg in result["messages"]:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                tool_steps.append({
                    "name": _attr(tc, "name"),
                    "args": _attr(tc, "args"),
                    "tool_call_id": _attr(tc, "id"),
                    "output": None
                })

        elif isinstance(msg, ToolMessage):
            for step in tool_steps:
                if step["tool_call_id"] == msg.tool_call_id:
                    step["output"] = msg.content
                    break

    tools_used = [step["name"] for step in tool_steps]

    return final_response, updated_history, tools_used, tool_steps
