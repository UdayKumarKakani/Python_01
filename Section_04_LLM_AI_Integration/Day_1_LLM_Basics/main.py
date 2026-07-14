"""
Day 1 - What is an LLM? (offline demos)
---------------------------------------
No API keys needed. Everything runs locally.

Shows three things:
1. How text is chopped into tokens (with tiktoken)
2. How much a page of text would cost to send to gpt-4o-mini
3. How the "temperature" dial reshapes a probability distribution

Run:
    python main.py
"""

import numpy as np
import tiktoken


def demo_tokens() -> None:
    print("--- 1. Tokens: how AI chops up words ---\n")
    enc = tiktoken.encoding_for_model("gpt-4o-mini")
    sentences = [
        "Hi.",
        "The quick brown fox jumps over the lazy dog.",
        "Unbelievable — this is a longer sentence.",
        "🎉🚀🎊 Emojis often cost more tokens than you'd expect!",
    ]
    for s in sentences:
        n = len(enc.encode(s))
        print(f"  {n:3d} tokens : {s}")
    print()


def demo_cost() -> None:
    print("--- 2. Cost of one page of text ---\n")
    enc = tiktoken.encoding_for_model("gpt-4o-mini")
    page = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 60

    tokens = len(enc.encode(page))
    # gpt-4o-mini input price ~ $0.15 per million tokens (early 2026)
    cost = (tokens / 1_000_000) * 0.15

    print(f"  tokens in one page      : {tokens}")
    print(f"  cost to send once       : ${cost:.6f}")
    print(f"  cost to send 1,000 times: ${cost*1000:.4f}")
    print()


def demo_temperature() -> None:
    print("--- 3. Temperature: the creativity dial ---\n")
    words = ["delicious", "tasty", "good", "ok", "awful"]
    logits = np.array([2.0, 1.5, 1.0, 0.2, -1.0])

    for T in (0.1, 1.0, 1.5):
        z = logits / T
        e = np.exp(z - z.max())
        probs = e / e.sum()
        print(f"  Temperature = {T}")
        for w, p in zip(words, probs):
            print(f"    {w:12s} {p:.2%}")
        print()


if __name__ == "__main__":
    demo_tokens()
    demo_cost()
    demo_temperature()
