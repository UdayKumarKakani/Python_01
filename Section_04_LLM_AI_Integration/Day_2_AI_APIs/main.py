"""
Day 2 - Using AI APIs (OpenAI + Claude)
---------------------------------------
Sends the same question to both providers and prints the replies side
by side, including token counts. Falls back gracefully if a key is missing.

Setup:
    Put OPENAI_API_KEY and ANTHROPIC_API_KEY in a .env file next to this script.

Run:
    python main.py
"""

import os

from dotenv import load_dotenv

load_dotenv()

QUESTION = "Explain a black hole to a 10-year-old in under 30 words."


def ask_openai() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        print("[openai] no key set — skipping.")
        return
    from openai import OpenAI
    client = OpenAI()

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a friendly assistant."},
            {"role": "user", "content": QUESTION},
        ],
    )
    print("--- OpenAI (gpt-4o-mini) ---")
    print(resp.choices[0].message.content)
    print(f"[tokens in={resp.usage.prompt_tokens}, out={resp.usage.completion_tokens}]\n")


def ask_claude() -> None:
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("[claude] no key set — skipping.")
        return
    from anthropic import Anthropic
    client = Anthropic()

    resp = client.messages.create(
        model="claude-3-5-haiku-latest",
        max_tokens=200,
        system="You are a friendly assistant.",
        messages=[{"role": "user", "content": QUESTION}],
    )
    print("--- Claude (3.5 Haiku) ---")
    print(resp.content[0].text)
    print(f"[tokens in={resp.usage.input_tokens}, out={resp.usage.output_tokens}]\n")


if __name__ == "__main__":
    print(f"Question: {QUESTION}\n")
    ask_openai()
    ask_claude()
