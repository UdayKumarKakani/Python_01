"""
Day 4 - Prompt Engineering
--------------------------
Three tiny demos on Together AI:

1. Weak prompt vs strong prompt on the same question.
2. Zero-shot vs "let's think step by step" on a math puzzle.
3. A hand-simulated ReAct trace answering a two-hop question.

Setup:
    Add TOGETHER_API_KEY to your .env file.

Run:
    python main.py
"""

import os
import sys

from dotenv import load_dotenv

load_dotenv()

if not os.getenv("TOGETHER_API_KEY"):
    sys.exit("Missing TOGETHER_API_KEY. Add it to .env and re-run.")

from together import Together

client = Together()
MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"


def ask(prompt: str, system: str | None = None,
        max_tokens: int = 300, temperature: float = 0.2) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    r = client.chat.completions.create(
        model=MODEL, messages=messages,
        max_tokens=max_tokens, temperature=temperature,
    )
    return r.choices[0].message.content.strip()


def demo_weak_vs_strong() -> None:
    print("--- 1. Weak vs strong prompt ---\n")
    snippet = (
        "The company reported Q3 revenue of $4.2B, up 12% year-over-year, "
        "driven by cloud services (+28%) and enterprise software (+9%), "
        "partially offset by declining hardware sales (-5%)."
    )
    weak = f"summarize this: {snippet}"
    strong = (
        "You are a financial analyst. Summarize the earnings snippet below "
        "in exactly 3 bullets, each under 12 words. No jargon. Highlight the "
        "biggest growth driver and the biggest drag.\n\n"
        f"Snippet: {snippet}"
    )
    print("[weak]\n",   ask(weak),   "\n")
    print("[strong]\n", ask(strong), "\n")


def demo_cot() -> None:
    print("--- 2. Chain of Thought ---\n")
    puzzle = (
        "A shop sells apples at 3 for $2 and oranges at 5 for $3. "
        "Priya buys 12 apples and 15 oranges and pays with a $20 bill. "
        "How much change does she get?"
    )
    print("[direct]",
          ask(puzzle + " Reply with just the dollar amount.", max_tokens=20))
    print("\n[step by step]\n",
          ask(puzzle + " Let's think step by step.", max_tokens=300))
    print("\n(correct answer is $3)")


def demo_react() -> None:
    print("\n--- 3. ReAct trace ---\n")
    system = (
        "You are a research agent. Answer using this exact format, one step at a time:\n"
        "Thought: ...\nAction: search(\"...\")\nObservation: <will be filled by tool>\n"
        "...repeat as needed...\nFinal Answer: <answer>\n"
        "Available actions: search(query). Stop after Final Answer."
    )
    seed = (
        "Question: What's the approximate population of the capital of France?\n"
        "Thought: I need to know what the capital of France is.\n"
        "Action: search(\"capital of France\")\n"
        "Observation: Paris\n"
        "Thought:"
    )
    print(ask(seed, system=system, max_tokens=400))


if __name__ == "__main__":
    demo_weak_vs_strong()
    demo_cot()
    demo_react()
