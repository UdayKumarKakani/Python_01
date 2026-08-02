"""
Day 2 - Function-calling agent with real tools
-----------------------------------------------
Requires TOGETHER_API_KEY.
Optional: TAVILY_API_KEY for real web search (mock is used otherwise).

Shows:
1. Tool schemas
2. calc, web_search, fetch_url implementations
3. The function-calling loop

Run:
    python main.py
"""

import ast
import json
import operator
import os

import httpx
from dotenv import load_dotenv
from together import Together

load_dotenv()
assert os.getenv("TOGETHER_API_KEY"), "Set TOGETHER_API_KEY in .env"
llm = Together()
MODEL = "openai/gpt-oss-20b"


_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub,
    ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.Pow: operator.pow, ast.USub: operator.neg,
}


def _eval(n):
    if isinstance(n, ast.Num):
        return n.n
    if isinstance(n, ast.BinOp):
        return _OPS[type(n.op)](_eval(n.left), _eval(n.right))
    if isinstance(n, ast.UnaryOp):
        return _OPS[type(n.op)](_eval(n.operand))
    raise ValueError("bad expr")


def calc(expr: str) -> str:
    try:
        return str(_eval(ast.parse(expr, mode="eval").body))
    except Exception as e:
        return f"error: {e}"


def web_search(query: str) -> str:
    if not os.getenv("TAVILY_API_KEY"):
        return f"[MOCK] top result for '{query}': (set TAVILY_API_KEY for real search)"
    from tavily import TavilyClient
    tv = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    r = tv.search(query, max_results=3)
    return "\n".join(f"- {x['title']}: {x['content'][:200]}" for x in r["results"])


def fetch_url(url: str) -> str:
    try:
        r = httpx.get(url, timeout=10, follow_redirects=True)
        r.raise_for_status()
        text = r.text
        return text[:2000] + ("... [truncated]" if len(text) > 2000 else "")
    except Exception as e:
        return f"error: {e}"


TOOLS = {"web_search": web_search, "fetch_url": fetch_url, "calc": calc}

TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "web_search",
        "description": "Search the web for recent information. Returns short snippets.",
        "parameters": {"type": "object",
                       "properties": {"query": {"type": "string"}},
                       "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "fetch_url",
        "description": "Fetch the plain-text content of a URL (truncated to 2 KB).",
        "parameters": {"type": "object",
                       "properties": {"url": {"type": "string"}},
                       "required": ["url"]}}},
    {"type": "function", "function": {
        "name": "calc",
        "description": "Evaluate a simple math expression like '(29*12)+100'.",
        "parameters": {"type": "object",
                       "properties": {"expr": {"type": "string"}},
                       "required": ["expr"]}}},
]


def agent(question: str, max_steps: int = 6) -> str:
    messages = [
        {"role": "system", "content":
            "You are a helpful research assistant. Use tools when useful. "
            "When you have the final answer, respond in plain text without tool calls."},
        {"role": "user", "content": question},
    ]
    for step in range(max_steps):
        resp = llm.chat.completions.create(
            model=MODEL, messages=messages,
            tools=TOOL_SCHEMAS, tool_choice="auto", temperature=0.0,
        )
        msg = resp.choices[0].message
        if msg.tool_calls:
            messages.append({
                "role": "assistant", "content": msg.content or "",
                "tool_calls": [tc.model_dump() for tc in msg.tool_calls],
            })
            for tc in msg.tool_calls:
                name = tc.function.name
                args = json.loads(tc.function.arguments)
                arg_val = next(iter(args.values()))
                obs = TOOLS[name](arg_val) if name in TOOLS else f"unknown tool {name}"
                print(f"\n--- step {step+1}: {name}({arg_val!r}) ---\n{obs[:300]}")
                messages.append({"role": "tool", "tool_call_id": tc.id,
                                 "name": name, "content": obs})
            continue
        return msg.content or "(empty)"
    return "(max steps reached)"


if __name__ == "__main__":
    q = "What is (29 * 12) + 100?"
    print(f"Q: {q}\n\n=== FINAL ===\n{agent(q)}")
