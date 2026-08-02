"""
Day 1 - ReAct agent from scratch
---------------------------------
Requires TOGETHER_API_KEY in .env.

Shows:
1. Two tools: safe calc, dict lookup
2. A ReAct loop parsing Thought/Action/Observation
3. stop-sequence + max_steps safety

Run:
    python main.py
"""

import ast
import operator
import os
import re

from dotenv import load_dotenv
from together import Together

load_dotenv()
assert os.getenv("TOGETHER_API_KEY"), "Set TOGETHER_API_KEY in .env"
llm = Together()


_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub,
    ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.Pow: operator.pow, ast.USub: operator.neg,
}


def _eval(node):
    if isinstance(node, ast.Num):
        return node.n
    if isinstance(node, ast.BinOp):
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp):
        return _OPS[type(node.op)](_eval(node.operand))
    raise ValueError("unsupported expression")


def calc(expr: str) -> str:
    try:
        return str(_eval(ast.parse(expr, mode="eval").body))
    except Exception as e:
        return f"error: {e}"


FACTS = {
    "acmecloud pro price": "$29 per month",
    "acmecloud free tier": "10 GB storage, 100 API calls/day",
    "acmecloud founders":  "Priya Rao and Marcus Chen, 2019",
}


def lookup(key: str) -> str:
    return FACTS.get(key.lower().strip(), "not found")


TOOLS = {"calc": calc, "lookup": lookup}
SYSTEM = (
    "You are a ReAct agent. On every turn respond with EXACTLY one of:\n"
    "Thought: <your reasoning>\n"
    "Action: calc[<math expression>]\n"
    "Action: lookup[<key to look up>]\n"
    "Action: finish[<final answer to the user>]\n\n"
    "Rules:\n"
    "- Emit ONE action per turn.\n"
    "- After you see Observation: text, decide what to do next.\n"
    "- When you have the final answer, use finish[...] and stop."
)
ACTION_RE = re.compile(r"Action:\s*(\w+)\[(.*?)\]", re.DOTALL)


def agent(question: str, max_steps: int = 6) -> str:
    transcript = [
        {"role": "system", "content": SYSTEM},
        {"role": "user",   "content": f"Question: {question}"},
    ]
    for step in range(max_steps):
        resp = llm.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=transcript,
            temperature=0.0,
            stop=["Observation:"],
        )
        turn = resp.choices[0].message.content.strip()
        print(f"\n--- step {step+1} ---\n{turn}")

        m = ACTION_RE.search(turn)
        if not m:
            return "(agent stopped - could not parse an Action)"
        name, arg = m.group(1), m.group(2).strip()

        if name == "finish":
            return arg
        if name not in TOOLS:
            return f"(unknown tool: {name})"

        obs = TOOLS[name](arg)
        print(f"Observation: {obs}")
        transcript.append({"role": "assistant", "content": turn})
        transcript.append({"role": "user",      "content": f"Observation: {obs}"})

    return "(agent stopped - max steps reached)"


if __name__ == "__main__":
    q = "If I pay for AcmeCloud Pro for 2 years, how much is that in total?"
    print(f"Q: {q}")
    print(f"\n=== FINAL ===\n{agent(q)}")
