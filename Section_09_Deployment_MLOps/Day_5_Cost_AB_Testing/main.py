"""
Day 5 - Cost tracking + budget cap demo
----------------------------------------
Standalone script - no external services.
"""

import sqlite3
import time

from rich.console import Console
from rich.table import Table


PRICES_USD_PER_MTOK = {
    "openai/gpt-oss-20b":                       {"in": 0.05, "out": 0.20},   # our default
    "openai/gpt-oss-120b":                      {"in": 0.15, "out": 0.60},   # bigger sibling
    "meta-llama/Llama-3.3-70B-Instruct-Turbo":  {"in": 0.88, "out": 0.88},
    "openai/gpt-4o-mini":                       {"in": 0.15, "out": 0.60},
}

conn = sqlite3.connect(":memory:")
conn.execute(
    "CREATE TABLE usage(ts INT, user TEXT, endpoint TEXT, model TEXT, "
    "in_tokens INT, out_tokens INT, cost_usd REAL)"
)


def log_call(user: str, endpoint: str, model: str, in_tok: int, out_tok: int) -> float:
    p = PRICES_USD_PER_MTOK[model]
    cost = (in_tok / 1e6) * p["in"] + (out_tok / 1e6) * p["out"]
    conn.execute("INSERT INTO usage VALUES (?, ?, ?, ?, ?, ?, ?)",
                 (int(time.time()), user, endpoint, model, in_tok, out_tok, cost))
    return cost


def demo() -> None:
    calls = [
        ("alice", "/ask", "openai/gpt-oss-20b",  500, 200),
        ("alice", "/ask", "openai/gpt-oss-20b",  700, 300),
        ("bob",   "/ask", "openai/gpt-oss-120b",                     1500, 400),
        ("bob",   "/ask", "openai/gpt-4o-mini",                       500, 200),
    ]
    for c in calls:
        log_call(*c)

    console = Console()
    t = Table(title="Today's cost per user")
    t.add_column("User");   t.add_column("Tokens", justify="right")
    t.add_column("USD", justify="right")
    for user, tok, usd in conn.execute(
        "SELECT user, SUM(in_tokens+out_tokens), ROUND(SUM(cost_usd),5) "
        "FROM usage GROUP BY user ORDER BY 3 DESC"
    ):
        t.add_row(user, str(tok), f"${usd:.5f}")
    console.print(t)


if __name__ == "__main__":
    demo()
