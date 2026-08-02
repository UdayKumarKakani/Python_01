"""
Day 3 - LangGraph stateful agent (plan -> act -> review)
---------------------------------------------------------
Requires TOGETHER_API_KEY in .env.

Shows:
1. TypedDict state
2. plan / act / review nodes
3. A conditional edge that retries on short answers

Run:
    python main.py
"""

import operator
import os
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph
from together import Together

load_dotenv()
assert os.getenv("TOGETHER_API_KEY"), "Set TOGETHER_API_KEY in .env"
llm = Together()
MODEL = "openai/gpt-oss-20b"


class AgentState(TypedDict):
    question: str
    plan: str
    steps_taken: Annotated[list[str], operator.add]
    answer: str


def _chat(prompt: str, temperature: float = 0.0) -> str:
    r = llm.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return r.choices[0].message.content.strip()


def plan_node(state: AgentState) -> dict:
    plan = _chat(f"Question: {state['question']}\nWrite a 1-2 sentence plan.")
    return {"plan": plan, "steps_taken": ["planned"]}


def act_node(state: AgentState) -> dict:
    ans = _chat(
        f"Question: {state['question']}\nPlan: {state['plan']}\nWrite the final answer.",
        temperature=0.2,
    )
    return {"answer": ans, "steps_taken": ["acted"]}


def review_node(state: AgentState) -> dict:
    ans = _chat(
        f"Question: {state['question']}\nAnswer: {state['answer']}\n"
        "Rewrite the answer to be crisp and under 50 words."
    )
    return {"answer": ans, "steps_taken": ["reviewed"]}


def is_answer_ok(state: AgentState) -> str:
    return "end" if len(state["answer"]) > 40 else "retry"


builder = StateGraph(AgentState)
builder.add_node("plan",   plan_node)
builder.add_node("act",    act_node)
builder.add_node("review", review_node)
builder.add_edge(START,   "plan")
builder.add_edge("plan",  "act")
builder.add_conditional_edges("act", is_answer_ok, {"end": "review", "retry": "plan"})
builder.add_edge("review", END)
graph = builder.compile()


if __name__ == "__main__":
    final = graph.invoke({"question": "Explain HTTPS in one paragraph."})
    print("Steps :", final["steps_taken"])
    print("\nPlan  :", final["plan"])
    print("\nAnswer:", final["answer"])
