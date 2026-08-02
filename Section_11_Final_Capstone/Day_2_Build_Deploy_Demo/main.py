"""
Section 11 - Day 2 helpers
---------------------------
Three things you'll want during the capstone build:

1. LangGraph agent starter (fill in tool bodies for your track)
2. Per-user daily budget cap FastAPI dependency
3. Cost report generator (reads your usage.db, prints a README-ready table)

Run:
    python main.py           # runs the cost report demo
    python main.py --agent   # runs the agent demo
"""

import operator
import sqlite3
import sys
from datetime import datetime
from typing import Annotated, TypedDict

from fastapi import Depends, HTTPException, status
from langgraph.graph import END, START, StateGraph
from rich.console import Console
from rich.table import Table


# ============================================================
# 1. Agent starter
# ============================================================
class State(TypedDict):
    question: str
    plan: str
    findings: list[str]
    answer: str
    steps: Annotated[list[str], operator.add]


def plan_node(s: State) -> dict:
    # TODO: LLM call to make a short plan
    return {"plan": "search kb then answer", "steps": ["planned"]}


def search_node(s: State) -> dict:
    # TODO: call your retrieve_tool(s['question'])
    return {"findings": [f"pretend chunk for {s['question']!r}"], "steps": ["searched"]}


def answer_node(s: State) -> dict:
    # TODO: LLM call with plan + findings
    return {"answer": f"(demo) answer to {s['question']!r}", "steps": ["answered"]}


def build_graph():
    b = StateGraph(State)
    b.add_node("plan", plan_node)
    b.add_node("search", search_node)
    b.add_node("answer", answer_node)
    b.add_edge(START, "plan")
    b.add_edge("plan", "search")
    b.add_edge("search", "answer")
    b.add_edge("answer", END)
    return b.compile()


# ============================================================
# 2. Budget-cap dependency
# ============================================================
DAILY_CAP_USD = 0.20


def _open_db(path: str = "usage.db"):
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.execute("""CREATE TABLE IF NOT EXISTS usage(
        ts TEXT, user TEXT, endpoint TEXT, model TEXT,
        in_tokens INT, out_tokens INT, cost_usd REAL
    )""")
    conn.commit()
    return conn


_conn = _open_db()


def log(user: str, endpoint: str, model: str, in_tok: int, out_tok: int, cost: float) -> None:
    _conn.execute(
        "INSERT INTO usage VALUES (?, ?, ?, ?, ?, ?, ?)",
        (datetime.utcnow().isoformat(), user, endpoint, model, in_tok, out_tok, cost),
    )
    _conn.commit()


def todays_spend(user: str) -> float:
    row = _conn.execute(
        "SELECT COALESCE(SUM(cost_usd),0) FROM usage "
        "WHERE user=? AND date(ts)=date('now')",
        (user,),
    ).fetchone()
    return row[0]


def check_budget_factory(current_user_dep):
    """Wrap your existing current_user dependency to chain them."""
    def check(user: str = Depends(current_user_dep)) -> str:
        if todays_spend(user) > DAILY_CAP_USD:
            raise HTTPException(
                status.HTTP_429_TOO_MANY_REQUESTS,
                f"Daily budget hit (${todays_spend(user):.2f})",
            )
        return user
    return check


# ============================================================
# 3. Cost report
# ============================================================
def report(db_path: str = "usage.db") -> None:
    conn = sqlite3.connect(db_path)
    row = conn.execute(
        "SELECT COUNT(*), COALESCE(SUM(in_tokens+out_tokens),0), "
        "COALESCE(SUM(cost_usd),0) FROM usage"
    ).fetchone()
    n, tokens, cost = row
    active_users = conn.execute("SELECT COUNT(DISTINCT user) FROM usage").fetchone()[0]

    days = _days_of_data(conn)

    t = Table(title="Capstone cost report")
    t.add_column("Metric"); t.add_column("Value", justify="right")
    t.add_row("Total requests",            f"{n:,}")
    t.add_row("Total tokens (in+out)",     f"{tokens:,}")
    t.add_row("Total cost USD",            f"${cost:.4f}")
    t.add_row("Distinct active users",     str(active_users))
    if n:
        t.add_row("Avg tokens per request", f"{tokens/n:.0f}")
        t.add_row("Avg cost per request",   f"${cost/n:.6f}")
    if active_users:
        t.add_row("Projected cost/user/month",
                  f"${(cost/active_users) * 30 / days:.4f}")

    Console().print(t)


def _days_of_data(conn) -> float:
    r = conn.execute("SELECT MIN(ts), MAX(ts) FROM usage").fetchone()
    if not r or not r[0]:
        return 1
    a, b = datetime.fromisoformat(r[0]), datetime.fromisoformat(r[1])
    return max(1, (b - a).total_seconds() / 86400)


if __name__ == "__main__":
    if "--agent" in sys.argv:
        g = build_graph()
        print(g.invoke({"question": "What is our refund policy?",
                        "plan": "", "findings": [], "answer": "", "steps": []}))
    else:
        # Seed some fake usage so the report has something to show
        for u, t, c in [("alice", 1200, 0.001), ("alice", 800, 0.0007), ("bob", 1500, 0.0012)]:
            log(u, "/rag/ask", "openai/gpt-oss-20b", t, t // 4, c)
        report()
